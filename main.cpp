#include <atomic>
#include <stdexcept>
#include <memory>
#include <type_traits>
#include <chrono>
#include <algorithm>
#include <thread>
#include <vector>
#include <iostream>
#include <pthread.h>
#include <sched.h>
#include <x86intrin.h>
#include <unistd.h>

template <typename T, typename Alloc = std::allocator<T>>
class spsc {
private:
	using allocator_traits = std::allocator_traits<Alloc>;
	Alloc alloc_;
	T* buffer_;
	const std::size_t size_;
	const std::size_t mask_;

	alignas(64) std::atomic<uint64_t> head_{0}; // read
	char pad1[64 - sizeof(std::atomic<uint64_t>)];	

	alignas(64) std::atomic<uint64_t> tail_{0}; // write
	char pad2[64 - sizeof(std::atomic<uint64_t>)];	

public:
	explicit spsc(size_t size_pow2, const Alloc& alloc = Alloc())
	: alloc_(alloc),
	  size_(size_pow2),
	  mask_(size_pow2-1)
	{
		if((size_pow2 & (size_pow2-1)) != 0) {
			throw std::invalid_argument("size must be a power of two");
		}

		buffer_ = allocator_traits::allocate(alloc_, size_);
		for(std::size_t i=0; i<size_; i++) {
			allocator_traits::construct(alloc_, buffer_+i);
		}
	}

	~spsc() {
		destroy_remaining();
		allocator_traits::deallocate(alloc_, buffer_, size_);
	}

	// Non-copyable, non-movable
	spsc(const spsc&) = delete;
	spsc& operator=(const spsc&) = delete;
	
	bool push(const T& value) noexcept {
		const auto tail = tail_.load(std::memory_order_relaxed);
		const auto head = head_.load(std::memory_order_acquire);

		if(((tail+1) & mask_) == (head & mask_)) {
			return false;
		}

		buffer_[tail & mask_] = value;

		tail_.store(tail+1, std::memory_order_release);
		return true;
	}

	bool pop(T& out) noexcept {
		const auto head = head_.load(std::memory_order_relaxed);
		const auto tail = tail_.load(std::memory_order_acquire);

		if(head == tail) {
			return false;
		}

		out = buffer_[head & mask_];
		
		head_.store(head+1, std::memory_order_release);
		return true;
	}

	bool empty() const noexcept {
		return head_.load(std::memory_order_acquire) ==
			tail_.load(std::memory_order_acquire);
	}

	bool full() const noexcept {
		const auto tail = tail_.load(std::memory_order_relaxed);
		const auto head = head_.load(std::memory_order_acquire);

		return ((tail+1) & mask_) == (head & mask_);
	}

	std::size_t capacity() const noexcept { return size_; }

private:
	void destroy_remaining() {
		if constexpr(!std::is_trivially_destructible_v<T>) {
			const auto head = head_.load(std::memory_order_relaxed);
			const auto tail = tail_.load(std::memory_order_relaxed);
			for(auto i=head; i!=tail; i++) {
				allocator_traits::destroy(alloc_, buffer_+(i & mask_));
			}
		}	
	}	
};


void pin_thread_to_core(int core_id) {
	cpu_set_t cpuset;
	CPU_ZERO(&cpuset);
	CPU_SET(core_id, &cpuset);
	pthread_setaffinity_np(pthread_self(), sizeof(cpu_set_t), &cpuset);
}

void set_realtime() {
	sched_param sch;
	sch.sched_priority = 80;
	pthread_setschedparam(pthread_self(), SCHED_FIFO, &sch);
}

inline uint64_t rdtsc_now() {
	unsigned int aux;
	_mm_lfence();
	auto t = __rdtscp(&aux);
	_mm_lfence();
	return t;
}

double calibrate_ghz(int samples = 5) {
    using clk = std::chrono::steady_clock;
    uint64_t c0 = rdtsc_now();
    auto t0 = clk::now();
    std::this_thread::sleep_for(std::chrono::milliseconds(100));
    uint64_t c1 = rdtsc_now();
    auto t1 = clk::now();
    double elapsed_s = std::chrono::duration<double>(t1 - t0).count();
    double hz = double(c1 - c0) / elapsed_s;
    return hz / 1e9; // GHz
}

double cycles_to_ns(uint64_t cycles, double ghz) {
	double freq_hz = ghz*1e9;
	double seconds = double(cycles)/ freq_hz;
	return seconds * 1e9;
}

int main() {
	const size_t n = 2000000;
	spsc<uint64_t> q(1024);

	std::vector<uint64_t> lat_cycles(n); // stores cycle

	auto t_start = std::chrono::steady_clock::now();

	std::thread producer([&]() {
		pin_thread_to_core(0);
		set_realtime();

		for(size_t i=0; i<n; i++) {
			uint64_t t0 = rdtsc_now();

			while(!q.push(t0)){
				_mm_pause(); // pause = low-power spin
			}
		}
	});

	std::thread consumer([&]() {
		pin_thread_to_core(1);
		set_realtime();

		for(size_t i=0; i<n; i++) {
			uint64_t stamp;
			while(!q.pop(stamp)){
				_mm_pause();
			}
			uint64_t t1 = rdtsc_now();
			lat_cycles[i] = t1 - stamp;
		}
	});
		
	producer.join();
	consumer.join();

	auto t_end = std::chrono::steady_clock::now();
	std::sort(lat_cycles.begin(), lat_cycles.end());

	auto pct = [&](double p){
		size_t idx = size_t(p * lat_cycles.size());
		return lat_cycles[std::min(idx, lat_cycles.size() - 1)];
	};

	double ghz = calibrate_ghz();
	std::cerr << "Detected CPU freq: " << ghz << " GHz\n";

	auto to_ns = [&](uint64_t cyc) {
		return cycles_to_ns(cyc, ghz);
	};

	std::cout << "------- SPSC Ring Buffer Benchmark -------\n";
	std::cout << "Total ops: " << n << "\n";
	std::cout << "P50 : "  << to_ns(pct(0.50))  << " ns\n";
	std::cout << "P90 : "  << to_ns(pct(0.90))  << " ns\n";
	std::cout << "P99 : "  << to_ns(pct(0.99))  << " ns\n";
	std::cout << "P999: "  << to_ns(pct(0.999)) << " ns\n";
	std::cout << "Min : "  << to_ns(lat_cycles.front()) << " ns\n";
	std::cout << "Max : "  << to_ns(lat_cycles.back())  << " ns\n";

	double avg = 0;
	for (auto v : lat_cycles) avg += v;
	avg /= lat_cycles.size();

	std::cout << "Average latency: " << to_ns(avg) << " ns\n";

	double wall_seconds =
		std::chrono::duration_cast<std::chrono::duration<double>>(t_end - t_start).count();

	std::cout << "Wall time: " << wall_seconds << " s\n";
        std::cout << "Throughput: " << double(n) / wall_seconds << " ops/sec\n";

	return 0;
}



