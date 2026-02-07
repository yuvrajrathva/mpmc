#pragma once // compiler will only process its content once

#include <atomic>
#include <cstddef>
#include <cstdint>
#include <new>
#include <type_traits>
#include <utility>
#include <cassert>
#include <stdexcept>
#include <memory>
#include <thread>
#include <vector>
#include <iostream>
#include <algorithm>
#include <chrono>
#include <x86intrin.h>
#include <pthread.h>
#include <sched.h>
#include <cstring>

template<typename T, typename Alloc = std::allocator<T>>
class mpmc
{
private:
	struct alignas(64) CellCtrl {
	    std::atomic<size_t> seq;
	};

	struct CellData {
	    alignas(alignof(T)) unsigned char storage_bytes[sizeof(T)];
	};

        static T* storage_ptr(CellData& d) noexcept {
        	return std::launder(reinterpret_cast<T*>(d.storage_bytes));
        }
	

	Alloc alloc_;
	CellCtrl* ctrl_;
	CellData* data_;
	const size_t size_;
	const size_t mask_;
	
	alignas(64) std::atomic<size_t> head_{0}; // read
	char pad2[64 - sizeof(std::atomic<size_t>)];

	alignas(64) std::atomic<size_t> tail_{0}; // write
	char pad3[64 - sizeof(std::atomic<size_t>)];

public:
	explicit mpmc(size_t size_pow2, const Alloc& alloc = Alloc())
	: alloc_(alloc),
	  ctrl_(nullptr),
	  data_(nullptr),
	  size_(size_pow2),
	  mask_(size_pow2-1)
	{
		if(((size_pow2 & (size_pow2-1)) != 0) || size_pow2 == 0) {
			throw std::invalid_argument("size must be a power of two");
		}

		ctrl_ = static_cast<CellCtrl*>(::operator new[](sizeof(CellCtrl) * size_));
		data_ = static_cast<CellData*>(::operator new[](sizeof(CellData) * size_));
		for(size_t i=0; i<size_; i++){
			new (&ctrl_[i].seq) std::atomic<size_t>(i);
		}		
	}

	~mpmc(){
		// slot should contain a contructed T
		if constexpr (!std::is_trivially_destructible_v<T>) {
			T temp;
			while(pop(temp)) {}
		}

		::operator delete[](ctrl_);
		::operator delete[](data_);
	}

	// Non-copyable, non-movable
	mpmc(const mpmc&) = delete;
	mpmc& operator=(const mpmc&) = delete;

	bool push(const T& val) noexcept {
		return emplace(val);
	}

	bool pop(T& out) noexcept {
		size_t pos = head_.load(std::memory_order_relaxed);
		while(true) {
			size_t idx = pos & mask_;
			CellCtrl& c = ctrl_[idx];
			size_t seq = c.seq.load(std::memory_order_acquire);
			ptrdiff_t diff = static_cast<ptrdiff_t>(seq) - static_cast<ptrdiff_t>(pos+1);
	
			if(__builtin_expect(diff == 0,1)) {
				// attempt to claim this slot
				if(head_.compare_exchange_weak(pos, pos+1, std::memory_order_relaxed, std::memory_order_relaxed)) {
					// now we own this slot
					T* src = storage_ptr(data_[idx]);
					out = std::move(*src);
					src->~T();

					c.seq.store(pos + size_, std::memory_order_release); // marks it empty
					return true;
				}
				continue;
				// CAS failed -> pos updated, retry immediately
			}
			else if(diff < 0) {
				// seq < head + 1 => slot is not yet produced
				return false;
			}
			else {
				// slot not yet for us (other consumer moved head forward)
				pos = head_.load(std::memory_order_relaxed);
			}
		}
	}

	size_t capacity() const noexcept { return size_; };

private:
	// internal emplace to avoid code duplication
	template<typename... Args>
	bool emplace(Args&&... args) noexcept {
		size_t pos = tail_.load(std::memory_order_relaxed);
		while(true) {
			size_t idx = pos & mask_;
			CellCtrl& c = ctrl_[idx];
			size_t seq = c.seq.load(std::memory_order_acquire);
			ptrdiff_t diff = static_cast<ptrdiff_t>(seq) - static_cast<ptrdiff_t>(pos);

			if(__builtin_expect(diff == 0,1)) {
				size_t old_pos = pos;
				if(tail_.compare_exchange_weak(pos, pos+1, std::memory_order_relaxed, std::memory_order_relaxed)) {
					T* dest = storage_ptr(data_[idx]);
					new (dest) T(std::forward<Args>(args)...);

					c.seq.store(pos+1, std::memory_order_release);
					return true;
				}
				continue;		
			}
			else if(diff < 0) {
				return false;
			}
			else {
				pos = tail_.load(std::memory_order_relaxed);
			}
		}
	}
	
};	


// ---------- helpers (pinning, rdtsc, calibrate) ----------
int cpu_count = std::thread::hardware_concurrency();

void pin_thread_to_core(int core_id) {
    cpu_set_t cpuset;
    CPU_ZERO(&cpuset);

    core_id = core_id%cpu_count;
    CPU_SET(core_id, &cpuset);   // wrap around

    int rc = pthread_setaffinity_np(pthread_self(), sizeof(cpu_set_t), &cpuset);
    if (rc != 0) std::cerr << "affinity failed: " << strerror(rc) << "\n";
}

inline uint64_t rdtsc_now() {
    unsigned int aux;
    return __rdtscp(&aux);
}

static inline uint64_t monotonic_raw_ns() {
    timespec ts;
    clock_gettime(CLOCK_MONOTONIC_RAW, &ts);
    return uint64_t(ts.tv_sec) * 1'000'000'000ull + ts.tv_nsec;
}

double calibrate_ghz() {
    // pin calibration thread
    pin_thread_to_core(0);

    constexpr int samples = 5;
    constexpr uint64_t duration_ns = 200'000'000; // 200 ms

    double freqs[samples];

    for (int i = 0; i < samples; ++i) {
        uint64_t t0_ns = monotonic_raw_ns();
        uint64_t c0 = rdtsc_now();

        // busy wait instead of sleep
        while (monotonic_raw_ns() - t0_ns < duration_ns) {
            _mm_pause();
        }

        uint64_t t1_ns = monotonic_raw_ns();
        uint64_t c1 = rdtsc_now();

        double elapsed_s = double(t1_ns - t0_ns) * 1e-9;
        freqs[i] = double(c1 - c0) / elapsed_s / 1e9;
    }

    std::sort(freqs, freqs + samples);
    return freqs[samples / 2]; // median
}

double cycles_to_ns(uint64_t cycles, double ghz) {
    double freq_hz = ghz * 1e9;
    double seconds = double(cycles) / freq_hz;
    return seconds * 1e9;
}

int main() {
    const size_t producers = 2;
    const size_t consumers = 2;
    const size_t items_per_producer = 1'000'000; // each producer will produce this many items
    constexpr size_t WARMUP_OPS_PER_THREAD = 50'000; // skip first N samples per consumer
    std::atomic<bool> start_flag{false};

    const size_t queue_size = 1024; // power of 2
    mpmc<uint64_t> q(queue_size);

    const size_t total_items = producers * items_per_producer;
    std::atomic<size_t> consumed_total{0};

    std::cerr << "Producers: " << producers << ", Consumers: " << consumers
              << ", Items/prod: " << items_per_producer << ", Queue sz: " << queue_size << "\n";


    std::vector<std::vector<uint64_t>> prod_lats(producers);
    std::vector<std::vector<uint64_t>> cons_lats(consumers);

    auto t_start = std::chrono::steady_clock::now();

    // producer threads
    std::vector<std::thread> producer_threads;
    producer_threads.reserve(producers);
    std::atomic<size_t> producers_done{0};
    for (size_t p = 0; p < producers; ++p) {
        producer_threads.emplace_back([p, &q, items_per_producer, &producers_done, &prod_lats, &start_flag]() {
            pin_thread_to_core(p==0?0:2);
	    // ---------- wait for synchronized start ----------
            while (!start_flag.load(std::memory_order_acquire)) {
            	_mm_pause();
            }

            // ---------- WARM-UP ----------
            for (size_t i = 0; i < WARMUP_OPS_PER_THREAD; ++i) {
            	uint64_t dummy = 0xDEADBEEF;
            	while (!q.push(dummy)) {
                	_mm_pause();
            	}
            }
	    // Small barrier to let consumers catch up
	    std::this_thread::sleep_for(std::chrono::milliseconds(10));

            // ---------- MEASURED PHASE ----------
	    prod_lats[p].reserve(items_per_producer);
            for (size_t i = 0; i < items_per_producer; ++i) {
                // externally spin until pushed (queue is non-blocking)
                uint64_t t0 = rdtsc_now();
                while (!q.push(t0)) {
                    _mm_pause();
		    t0 = rdtsc_now();
                }
		uint64_t t1 = rdtsc_now();

		prod_lats[p].push_back(t1-t0);
            }

	    size_t done_count = producers_done.fetch_add(1, std::memory_order_release);
        });
    }

    // create and start consumer threads
    std::vector<std::thread> consumer_threads;
    consumer_threads.reserve(consumers);
    for (size_t c = 0; c < consumers; ++c) {
        consumer_threads.emplace_back([c, &q, &cons_lats, &producers_done, producers, items_per_producer, &start_flag, &consumed_total]() {
		pin_thread_to_core(c==0?1:3);		

		while (!start_flag.load(std::memory_order_acquire)) {
		    _mm_pause();
		}


		// ---------- WARM-UP ----------
		uint64_t warmup_item;
		size_t warmup_drained = 0;
	        size_t expected_warmup = (producers * WARMUP_OPS_PER_THREAD) / consumers;
    
		while (warmup_drained < expected_warmup) {
		        if (q.pop(warmup_item)) {
		            warmup_drained++;
		        } else {
		            _mm_pause();
		        }
		}
    
	        std::this_thread::sleep_for(std::chrono::milliseconds(10));

		// ---------- MEASURED PHASE ----------
		cons_lats[c].reserve((producers * items_per_producer) / consumers + 1024);

		size_t local_received = 0;
		uint64_t stamp;
		while (true) {
		    
		    uint64_t t0 = rdtsc_now();
		    if (q.pop(stamp)) {

			uint64_t t1 = rdtsc_now();
			cons_lats[c].push_back(t1-t0);
			consumed_total.fetch_add(1, std::memory_order_relaxed);
		    } else {
			// If everyone has already consumed the total, exit; otherwise spin.
			// Use acquire to observe the latest value.
			if (producers_done.load(std::memory_order_acquire) == producers) {
					break;
			}
			_mm_pause();
		    }
		}
	});
    }
    
    // allow all threads to start and pin
    std::this_thread::sleep_for(std::chrono::milliseconds(100));
    // synchronized start: warm-up + measurement begin
    start_flag.store(true, std::memory_order_release);
 
    // wait for completion
    for (auto &t : producer_threads) t.join();
    for (auto &t : consumer_threads) t.join();

    auto t_end = std::chrono::steady_clock::now();

    // gather all latencies
    std::vector<uint64_t> all_prod, all_cons;
    for (auto &v : prod_lats) all_prod.insert(all_prod.end(), v.begin(), v.end());
    for (auto &v : cons_lats) all_cons.insert(all_cons.end(), v.begin(), v.end());

    if (all_prod.empty() || all_cons.empty()) {
        std::cerr << "No latency samples collected.\n";
        return 1;
    }

    double ghz = calibrate_ghz();
    std::cerr << "Invariant TSC freq: " << ghz << " GHz\n";

    auto to_ns = [&](uint64_t cyc) { return cycles_to_ns(cyc, ghz); };

    std::cout << "------- MPMC Ring Buffer Benchmark -------\n";
    std::cout << "Total produced: " << all_prod.size() << "\n";
    std::cout << "Total consumed: " << all_cons.size() << "\n";

    auto report = [&](const char* name, std::vector<uint64_t>& v) {
	    std::sort(v.begin(), v.end());
//	    auto it = std::partition_point(v.begin(), v.end(), 
//                                   [](uint64_t x) { return x < 1200; }); // ~1000ns
//	    size_t filtered_count = std::distance(v.begin(), it);
    
    	    std::cout << "---- " << name << " ----\n";
    	    std::cout << "Samples: " << v.size() <<"\n";
	    auto pct = [&](double p) {
		size_t idx = size_t(p * v.size());
		return v[std::min(idx, v.size() - 1)];
	    };

//	    std::cout << "---- " << name << " ----\n";
	    std::cout << "P50 : "  << cycles_to_ns(pct(0.50), ghz) << " ns\n";
	    std::cout << "P90 : "  << cycles_to_ns(pct(0.90), ghz) << " ns\n";
	    std::cout << "P99 : "  << cycles_to_ns(pct(0.99), ghz) << " ns\n";
	    std::cout << "P99.9 : " << cycles_to_ns(pct(0.999), ghz) << " ns\n";
	    std::cout << "P99.99: " << cycles_to_ns(pct(0.9999), ghz) << " ns\n";
	    std::cout << "Min : "  << cycles_to_ns(v.front(), ghz) << " ns\n";
	    std::cout << "Max : "  << cycles_to_ns(v.back(), ghz) << " ns\n";

//	    double sum = 0;
//    	    for (size_t i = 0; i < filtered_count; ++i) {
//        	sum += v[i];
//    	    }
//    	    std::cout << "Avg (filtered): " << cycles_to_ns(sum/filtered_count, ghz) << " ns\n";
    };

    report("Enqueue cost", all_prod);
    report("Dequeue cost", all_cons);

    double wall_seconds = std::chrono::duration_cast<std::chrono::duration<double>>(t_end - t_start).count();
    std::cout << "Wall time: " << wall_seconds << " s\n";
    std::cout << "Throughput: " << double(total_items) / wall_seconds << " ops/sec\n";

    return 0;
}

