// We modify the memory management of seal for the adaption of GPU memory management
// The original license of seal is as follows:
// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT license.

#include "mempool.cuh"
#include <cmath>
#include <numeric>
#include <stdexcept>

using namespace std;

namespace phantom::util {
    MemoryPoolHeadMT::MemoryPoolHeadMT(size_t item_byte_count, bool clear_on_destruction)
            : clear_on_destruction_(clear_on_destruction), locked_(false), item_byte_count_(item_byte_count),
              item_count_(MemoryPool::first_alloc_count), first_item_(nullptr) {
        if ((item_byte_count_ == 0) || (item_byte_count_ > MemoryPool::max_batch_alloc_byte_count) ||
            (arith::mul_safe(item_byte_count_, MemoryPool::first_alloc_count) >
             MemoryPool::max_batch_alloc_byte_count)) {
            throw invalid_argument("invalid allocation size");
        }

        // Initial allocation
        allocation new_alloc;
        try {
            // CUDA memory allocation of size first_alloc_count * item_byte_count_
            cudaMalloc(&new_alloc.data_ptr, arith::mul_safe(MemoryPool::first_alloc_count, item_byte_count_));
        }
        catch (const bad_alloc &) {
            // Allocation failed; rethrow
            throw;
        }

        new_alloc.size = MemoryPool::first_alloc_count;
        new_alloc.free = MemoryPool::first_alloc_count;
        new_alloc.head_ptr = new_alloc.data_ptr;
        allocs_.clear();
        allocs_.push_back(new_alloc);
    }

    MemoryPoolHeadMT::~MemoryPoolHeadMT() noexcept {
        bool expected = false;
        while (!locked_.compare_exchange_strong(expected, true, memory_order_acquire)) {
            expected = false;
        }

        // Delete the items (but not the memory)
        MemoryPoolItem *curr_item = first_item_;
        while (curr_item) {
            MemoryPoolItem *next_item = curr_item->next();
            delete curr_item;
            curr_item = next_item;
        }
        first_item_ = nullptr;

        // Do we need to clear the memory?
        if (clear_on_destruction_) {
            // Delete the memory
            for (auto &alloc: allocs_) {
                size_t curr_alloc_byte_count = arith::mul_safe(item_byte_count_, alloc.size);
                cudaMemset(alloc.data_ptr, 0, curr_alloc_byte_count);

                // Delete this allocation
                cudaFree(alloc.data_ptr);
            }
        } else {
            // Delete the memory
            for (auto &alloc: allocs_) {
                // Delete this allocation
                cudaFree(alloc.data_ptr);
            }
        }

        allocs_.clear();
    }

    MemoryPoolHeadST::MemoryPoolHeadST(size_t item_byte_count, bool clear_on_destruction)
            : clear_on_destruction_(clear_on_destruction), item_byte_count_(item_byte_count),
              item_count_(MemoryPool::first_alloc_count), first_item_(nullptr) {
        if ((item_byte_count_ == 0) || (item_byte_count_ > MemoryPool::max_batch_alloc_byte_count) ||
            (arith::mul_safe(item_byte_count_, MemoryPool::first_alloc_count) >
             MemoryPool::max_batch_alloc_byte_count)) {
            throw invalid_argument("invalid allocation size");
        }

        // Initial allocation
        allocation new_alloc;
        try {
            cudaMalloc(&new_alloc.data_ptr, arith::mul_safe(MemoryPool::first_alloc_count, item_byte_count_));
        }
        catch (const bad_alloc &) {
            // Allocation failed; rethrow
            throw;
        }

        new_alloc.size = MemoryPool::first_alloc_count;
        new_alloc.free = MemoryPool::first_alloc_count;
        new_alloc.head_ptr = new_alloc.data_ptr;
        allocs_.clear();
        allocs_.push_back(new_alloc);
    }

    MemoryPoolHeadST::~MemoryPoolHeadST() noexcept {
        // Delete the items (but not the memory)
        MemoryPoolItem *curr_item = first_item_;
        while (curr_item) {
            MemoryPoolItem *next_item = curr_item->next();
            delete curr_item;
            curr_item = next_item;
        }
        first_item_ = nullptr;

        // Do we need to clear the memory?
        if (clear_on_destruction_) {
            // Delete the memory
            for (auto &alloc: allocs_) {
                size_t curr_alloc_byte_count = arith::mul_safe(item_byte_count_, alloc.size);

                // Delete this allocation
                cudaFree(alloc.data_ptr);
            }
        } else {
            // Delete the memory
            for (auto &alloc: allocs_) {
                // Delete this allocation
                cudaFree(alloc.data_ptr);
            }
        }

        allocs_.clear();
    }

    MemoryPoolItem *MemoryPoolHeadST::get() {
        MemoryPoolItem *old_first = first_item_;

        // Is pool empty?
        if (old_first == nullptr) {
            allocation &last_alloc = allocs_.back();
            MemoryPoolItem *new_item = nullptr;
            if (last_alloc.free > 0) {
                // Pool is empty; there is memory
                new_item = new MemoryPoolItem(last_alloc.head_ptr);
                last_alloc.free--;
                last_alloc.head_ptr += item_byte_count_;
            } else {
                // Pool is empty; there is no memory
                allocation new_alloc;

                // Increase allocation size unless we are already at max
                auto new_size = size_t(
                        ceil(MemoryPool::alloc_size_multiplier * static_cast<double>(last_alloc.size)));
                size_t new_alloc_byte_count = arith::mul_safe(new_size, item_byte_count_);
                if (new_alloc_byte_count > MemoryPool::max_batch_alloc_byte_count) {
                    new_size = last_alloc.size;
                    new_alloc_byte_count = new_size * item_byte_count_;
                }

                try {
                    cudaMalloc(&new_alloc.data_ptr, new_alloc_byte_count);
                }
                catch (const bad_alloc &) {
                    // Allocation failed; rethrow
                    throw;
                }

                new_alloc.size = new_size;
                new_alloc.free = new_size - 1;
                new_alloc.head_ptr = new_alloc.data_ptr + item_byte_count_;
                allocs_.push_back(new_alloc);
                item_count_ += new_size;
                new_item = new MemoryPoolItem(new_alloc.data_ptr);
            }

            return new_item;
        }

        // Pool is not empty
        first_item_ = old_first->next();
        old_first->next() = nullptr;
        return old_first;
    }

    MemoryPoolItem *MemoryPoolHeadMT::get() {
        bool expected = false;
        while (!locked_.compare_exchange_strong(expected, true, memory_order_acquire)) {
            expected = false;
        }
        MemoryPoolItem *old_first = first_item_;

        // Is pool empty?
        if (old_first == nullptr) {
            allocation &last_alloc = allocs_.back();
            MemoryPoolItem *new_item = nullptr;
            if (last_alloc.free > 0) {
                // Pool is empty; there is memory
                new_item = new MemoryPoolItem(last_alloc.head_ptr);
                last_alloc.free--;
                last_alloc.head_ptr += item_byte_count_; // we need to malloc one block of this size
            } else {
                // Pool is empty; there is no memory. So, we malloc a new allocation (physical malloc)
                allocation new_alloc;

                // Increase allocation size unless we are already at max
                auto new_size = size_t(
                        ceil(MemoryPool::alloc_size_multiplier * static_cast<double>(last_alloc.size)));
                size_t new_alloc_byte_count = arith::mul_safe(new_size, item_byte_count_);
                if (new_alloc_byte_count > MemoryPool::max_batch_alloc_byte_count) {
                    new_size = last_alloc.size;
                    new_alloc_byte_count = new_size * item_byte_count_;
                }

                try {
                    cudaMalloc(&new_alloc.data_ptr, new_alloc_byte_count);
                }
                catch (const bad_alloc &) {
                    // Allocation failed; rethrow
                    throw;
                }

                new_alloc.size = new_size;
                new_alloc.free = new_size - 1;
                new_alloc.head_ptr = new_alloc.data_ptr + item_byte_count_;
                allocs_.push_back(new_alloc);
                item_count_ += new_size;
                new_item = new MemoryPoolItem(new_alloc.data_ptr);
            }

            locked_.store(false, memory_order_release);
            return new_item;
        }

        // Pool is not empty
        first_item_ = old_first->next();
        old_first->next() = nullptr;
        locked_.store(false, memory_order_release);
        return old_first;
    }

    const size_t MemoryPool::max_single_alloc_byte_count = []() -> size_t {
        int bit_shift = static_cast<int>(ceil(log2(MemoryPool::alloc_size_multiplier)));
        if (bit_shift < 0 ||
            arith::unsigned_geq(bit_shift, sizeof(size_t) * static_cast<size_t>(arith::bits_per_byte))) {
            throw logic_error("alloc_size_multiplier too large");
        }
        return numeric_limits<size_t>::max() >> bit_shift;
    }();

    const size_t MemoryPool::max_batch_alloc_byte_count = []() -> size_t {
        int bit_shift = static_cast<int>(ceil(log2(MemoryPool::alloc_size_multiplier)));
        if (bit_shift < 0 ||
            arith::unsigned_geq(bit_shift, sizeof(size_t) * static_cast<size_t>(arith::bits_per_byte))) {
            throw logic_error("alloc_size_multiplier too large");
        }
        return numeric_limits<size_t>::max() >> bit_shift;
    }();

    MemoryPoolMT::~MemoryPoolMT() noexcept = default;

    Pointer<uint8_t> MemoryPoolMT::get_for_byte_count(size_t byte_count) {
        if (byte_count > max_single_alloc_byte_count) {
            throw invalid_argument("invalid allocation size");
        } else if (byte_count == 0) {
            return {};
        }

        /*uint8_t *pointer;
        cudaMalloc(&pointer, byte_count);
        return Pointer<uint8_t>(pointer, false);*/

        // Attempt to find size.
        ReaderLock reader_lock(pools_locker_.acquire_read());
        size_t start = 0;
        size_t end = pools_.size();
        while (start < end) {
            size_t mid = (start + end) / 2;
            MemoryPoolHead *mid_head = pools_[mid];
            size_t mid_byte_count = mid_head->item_byte_count();
            if (byte_count < mid_byte_count) {
                start = mid + 1;
            } else if (byte_count > mid_byte_count) {
                end = mid;
            } else {
                return Pointer<uint8_t>(mid_head);
            }
        }
        reader_lock.unlock();

        // Size was not found, so obtain an exclusive lock and search again.
        WriterLock writer_lock(pools_locker_.acquire_write());
        start = 0;
        end = pools_.size();
        while (start < end) {
            size_t mid = (start + end) / 2;
            MemoryPoolHead *mid_head = pools_[mid];
            size_t mid_byte_count = mid_head->item_byte_count();
            if (byte_count < mid_byte_count) {
                start = mid + 1;
            } else if (byte_count > mid_byte_count) {
                end = mid;
            } else {
                return Pointer<uint8_t>(mid_head);
            }
        }

        // Size was still not found, but we own an exclusive lock so just add it,
        // but first check if we are at maximum pool head count already.
        if (pools_.size() >= max_pool_head_count) {
            throw runtime_error("maximum pool head count reached");
        }

        MemoryPoolHead *new_head = new MemoryPoolHeadMT(byte_count, clear_on_destruction_);
        if (!pools_.empty()) {
            pools_.insert(pools_.begin() + static_cast<ptrdiff_t>(start), new_head);
        } else {
            pools_.emplace_back(new_head);
        }

        return Pointer<uint8_t>(new_head);
    }

    size_t MemoryPoolMT::alloc_byte_count() const {
        ReaderLock lock(pools_locker_.acquire_read());

        return accumulate(pools_.cbegin(), pools_.cend(), size_t(0), [](size_t byte_count, MemoryPoolHead *head) {
            return arith::add_safe(byte_count, arith::mul_safe(head->item_count(), head->item_byte_count()));
        });
    }

    MemoryPoolST::~MemoryPoolST() noexcept = default;

    Pointer<uint8_t> MemoryPoolST::get_for_byte_count(size_t byte_count) {
        if (byte_count > MemoryPool::max_single_alloc_byte_count) {
            throw invalid_argument("invalid allocation size");
        } else if (byte_count == 0) {
            return {};
        }

        // Attempt to find size.
        size_t start = 0;
        size_t end = pools_.size();
        while (start < end) {
            size_t mid = (start + end) / 2;
            MemoryPoolHead *mid_head = pools_[mid];
            size_t mid_byte_count = mid_head->item_byte_count();
            if (byte_count < mid_byte_count) {
                start = mid + 1;
            } else if (byte_count > mid_byte_count) {
                end = mid;
            } else {
                return Pointer<uint8_t>(mid_head);
            }
        }

        // Size was not found so just add it, but first check if we are at
        // maximum pool head count already.
        if (pools_.size() >= max_pool_head_count) {
            throw runtime_error("maximum pool head count reached");
        }

        MemoryPoolHead *new_head = new MemoryPoolHeadST(byte_count, clear_on_destruction_);
        if (!pools_.empty()) {
            pools_.insert(pools_.begin() + static_cast<ptrdiff_t>(start), new_head);
        } else {
            pools_.emplace_back(new_head);
        }

        return Pointer<uint8_t>(new_head);
    }

    size_t MemoryPoolST::alloc_byte_count() const {
        return accumulate(pools_.cbegin(), pools_.cend(), size_t(0), [](size_t byte_count, MemoryPoolHead *head) {
            return arith::add_safe(byte_count, arith::mul_safe(head->item_count(), head->item_byte_count()));
        });
    }
}
