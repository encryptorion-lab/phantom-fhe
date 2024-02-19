#pragma once

#include "util/defines.h"
#include "util/globals.h"
#include "lock.cuh"
#include <algorithm>
#include <atomic>
#include <cstdint>
#include <cstring>
#include <limits>
#include <memory>
#include <new>
#include <stdexcept>
#include <type_traits>
#include <vector>
#include <cmath>
#include <numeric>

namespace phantom::util {
    template<typename T = void, typename = std::enable_if_t<std::is_standard_layout<T>::value>>
    class Pointer;

    class MemoryPoolItem {
    public:
        explicit MemoryPoolItem(uint8_t *data) noexcept: data_(data) {
        }

        MemoryPoolItem &operator=(const MemoryPoolItem &assign) = delete;

        MemoryPoolItem(const MemoryPoolItem &copy) = delete;

        [[nodiscard]] inline uint8_t *data() noexcept {
            return data_;
        }

        [[nodiscard]] inline const uint8_t *data() const noexcept {
            return data_;
        }

        [[nodiscard]] inline MemoryPoolItem *&next() noexcept {
            return next_;
        }

        [[nodiscard]] inline const MemoryPoolItem *next() const noexcept {
            return next_;
        }

    private:
        uint8_t *data_ = nullptr;

        MemoryPoolItem *next_ = nullptr;
    };

    class MemoryPoolHead {
    public:
        struct allocation {
            allocation() : size(0), data_ptr(nullptr), free(0), head_ptr(nullptr) {
            }

            // Size of the allocation (number of items it can hold)
            size_t size;

            // Pointer to start of the allocation
            uint8_t *data_ptr;

            // How much free space is left (number of items that still fit)
            std::size_t free;

            // Pointer to current head of allocation
            uint8_t *head_ptr;
        };

        // The overriding functions are noexcept(false)
        virtual ~MemoryPoolHead() = default;

        // Byte size of the allocations (items) owned by this pool
        [[nodiscard]] virtual std::size_t item_byte_count() const noexcept = 0;

        // Total number of items allocated
        [[nodiscard]] virtual std::size_t item_count() const noexcept = 0;

        virtual MemoryPoolItem *get() = 0;

        // Return item back to this pool
        virtual void add(MemoryPoolItem *new_first) noexcept = 0;
    };

    class MemoryPoolHeadMT : public MemoryPoolHead {
    public:
        // Creates a new MemoryPoolHeadMT with allocation for one single item.
        explicit MemoryPoolHeadMT(std::size_t item_byte_count, bool clear_on_destruction = false);

        MemoryPoolHeadMT(const MemoryPoolHeadMT &copy) = delete;

        ~MemoryPoolHeadMT() noexcept override;

        MemoryPoolHeadMT &operator=(const MemoryPoolHeadMT &assign) = delete;

        // Byte size of the allocations (items) owned by this pool
        [[nodiscard]] inline std::size_t item_byte_count() const noexcept override {
            return item_byte_count_;
        }

        // Returns the total number of items allocated
        [[nodiscard]] inline std::size_t item_count() const noexcept override {
            return item_count_;
        }

        MemoryPoolItem *get() override;

        inline void add(MemoryPoolItem *new_first) noexcept override {
            bool expected = false;
            while (!locked_.compare_exchange_strong(expected, true, std::memory_order_acquire)) {
                expected = false;
            }
            MemoryPoolItem *old_first = first_item_;
            new_first->next() = old_first;
            first_item_ = new_first;
            locked_.store(false, std::memory_order_release);
        }

    private:
        const bool clear_on_destruction_;

        mutable std::atomic<bool> locked_;

        const std::size_t item_byte_count_;

        // item_count elements, each has size "item_byte_count_"
        std::size_t item_count_;

        // use vector to maintain the physically allocated memory
        std::vector<allocation> allocs_;

        // have been malloced, assigned, and released, therefor can be directly used next time
        MemoryPoolItem *volatile first_item_;
    };

    class MemoryPoolHeadST : public MemoryPoolHead {
    public:
        // Creates a new MemoryPoolHeadST with allocation for one single item.
        explicit MemoryPoolHeadST(std::size_t item_byte_count, bool clear_on_destruction = false);

        MemoryPoolHeadST(const MemoryPoolHeadST &copy) = delete;

        ~MemoryPoolHeadST() noexcept override;

        MemoryPoolHeadST &operator=(const MemoryPoolHeadST &assign) = delete;

        // Byte size of the allocations (items) owned by this pool
        [[nodiscard]] inline std::size_t item_byte_count() const noexcept override {
            return item_byte_count_;
        }

        // Returns the total number of items allocated
        [[nodiscard]] inline std::size_t item_count() const noexcept override {
            return item_count_;
        }

        [[nodiscard]] MemoryPoolItem *get() override;

        inline void add(MemoryPoolItem *new_first) noexcept override {
            new_first->next() = first_item_;
            first_item_ = new_first;
        }

    private:
        const bool clear_on_destruction_;

        std::size_t item_byte_count_;

        std::size_t item_count_;

        std::vector<allocation> allocs_;

        MemoryPoolItem *first_item_;
    };

    class MemoryPool {
    public:
        static constexpr double alloc_size_multiplier = 1.05; //can only be 1, otherwise Memasyncattach will fail

        // Largest size of single allocation that can be requested from memory pool
        static const std::size_t max_single_alloc_byte_count;

        // One memory pool can be used to maintain "max_pool_head_count" number of pools
        static constexpr std::size_t max_pool_head_count = (std::numeric_limits<std::size_t>::max)();

        // Largest allowed size of batch allocation (physicall malloc, to reduce the malloc number)
        static const std::size_t max_batch_alloc_byte_count;

        // first only malloc "first_alloc_count" items
        static constexpr std::size_t first_alloc_count = 1;

        virtual ~MemoryPool() = default;

        virtual Pointer<uint8_t> get_for_byte_count(std::size_t byte_count) = 0;

        [[nodiscard]] virtual std::size_t pool_count() const = 0;

        [[nodiscard]] virtual std::size_t alloc_byte_count() const = 0;
    };

    class MemoryPoolMT : public MemoryPool {
    public:
        explicit MemoryPoolMT(bool clear_on_destruction = false) : clear_on_destruction_(clear_on_destruction) {
        };

        MemoryPoolMT(const MemoryPoolMT &copy) = delete;

        ~MemoryPoolMT() noexcept override;

        MemoryPoolMT &operator=(const MemoryPoolMT &assign) = delete;

        [[nodiscard]] Pointer<uint8_t> get_for_byte_count(std::size_t byte_count) override;

        [[nodiscard]] inline std::size_t pool_count() const override {
            ReaderLock lock(pools_locker_.acquire_read());
            return pools_.size();
        }

        [[nodiscard]] std::size_t alloc_byte_count() const override;

        void inline Release() noexcept {
            WriterLock lock(pools_locker_.acquire_write());
            for (MemoryPoolHead *head: pools_) {
                delete head;
            }
            pools_.clear();
        }

    protected:
        const bool clear_on_destruction_;

        mutable ReaderWriterLocker pools_locker_;

        // each pool corresponding for one size memory
        std::vector<MemoryPoolHead *> pools_;
    };

    class MemoryPoolST : public MemoryPool {
    public:
        explicit MemoryPoolST(bool clear_on_destruction = false) : clear_on_destruction_(clear_on_destruction) {
        };

        MemoryPoolST(const MemoryPoolMT &copy) = delete;

        ~MemoryPoolST() noexcept override;

        MemoryPoolST &operator=(const MemoryPoolST &assign) = delete;

        [[nodiscard]] Pointer<uint8_t> get_for_byte_count(std::size_t byte_count) override;

        [[nodiscard]] inline std::size_t pool_count() const override {
            return pools_.size();
        }

        [[nodiscard]] std::size_t alloc_byte_count() const override;

        void inline Release() noexcept {
            for (MemoryPoolHead *head: pools_) {
                delete head;
            }
            pools_.clear();
        }

    protected:
        const bool clear_on_destruction_;

        // each pool corresponding for one size memory
        std::vector<MemoryPoolHead *> pools_;
    };

    template<>
    class Pointer<uint8_t> {
        friend class MemoryPoolMT;

        friend class MemoryPoolST;

    public:
        template<typename, typename>
        friend
        class Pointer;

        Pointer() = default;

        // Move of the same type
        Pointer(Pointer<uint8_t> &&source) noexcept
            : data_(source.data_), head_(source.head_), item_(source.item_), alias_(source.alias_) {
            source.data_ = nullptr;
            source.head_ = nullptr;
            source.item_ = nullptr;
            source.alias_ = false;
        }

        Pointer(const Pointer<uint8_t> &copy) = delete;

        inline auto &operator=(Pointer<uint8_t> &&assign) noexcept {
            acquire(std::move(assign));
            return *this;
        }

        Pointer<uint8_t> &operator=(const Pointer<uint8_t> &assign) = delete;

        __host__ __device__ inline uint8_t *get() const noexcept {
            return data_;
        }

        [[nodiscard]] inline uint8_t *operator->() const noexcept {
            return data_;
        }

        [[nodiscard]] inline bool is_alias() const noexcept {
            return alias_;
        }

        // This is used to replace cudaFree(), for adding the memory of destroyed object to the pool
        inline void release() noexcept {
            if (head_) {
                // Return the memory to pool
                head_->add(item_);
            }
            else if (data_ && !alias_) {
                // Free the memory
                cudaFree(data_);
            }

            data_ = nullptr;
            head_ = nullptr;
            item_ = nullptr;
            alias_ = false;
        }

        void acquire(Pointer<uint8_t> &other) noexcept {
            if (this == &other) {
                return;
            }

            release();

            data_ = other.data_;
            head_ = other.head_;
            item_ = other.item_;
            alias_ = other.alias_;
            other.data_ = nullptr;
            other.head_ = nullptr;
            other.item_ = nullptr;
            other.alias_ = false;
        }

        inline void acquire(Pointer &&other) noexcept {
            acquire(other);
        }

        ~Pointer() noexcept {
            release();
        }

        [[nodiscard]] inline static Pointer<uint8_t> Owning(uint8_t *pointer) noexcept {
            return {pointer, false};
        }

        [[nodiscard]] inline static auto Aliasing(uint8_t *pointer) noexcept -> Pointer<uint8_t> {
            return {pointer, true};
        }

    private:
        Pointer(uint8_t *pointer, bool alias) noexcept: data_(pointer), alias_(alias) {
        }

        explicit Pointer(class MemoryPoolHead *head) {
            if (!head) {
                throw std::invalid_argument("head cannot be null");
            }
            head_ = head;
            item_ = head->get();
            data_ = item_->data();
        }

        uint8_t *data_ = nullptr;

        MemoryPoolHead *head_ = nullptr;

        MemoryPoolItem *item_ = nullptr;

        bool alias_ = false;
    };

    template<typename T, typename>
    class Pointer {
        friend class MemoryPoolMT;

        friend class MemoryPoolST;

    public:
        friend class Pointer<uint8_t>;

        Pointer() {
            data_ = nullptr;
            head_ = nullptr;
            item_ = nullptr;
            alias_ = false;
        }

        // Move of the same type
        explicit Pointer(Pointer<T> &&source) noexcept
            : data_(source.data_), head_(source.head_), item_(source.item_), alias_(source.alias_) {
            source.data_ = nullptr;
            source.head_ = nullptr;
            source.item_ = nullptr;
            source.alias_ = false;
        }

        // Move when T is not seal_byte
        template<typename... Args>
        explicit Pointer(Pointer<uint8_t> &&source, Args &&... args) {
            // Cannot acquire a non-pool pointer of different type
            if (!source.head_ && source.data_) {
                throw std::invalid_argument("cannot acquire a non-pool pointer of different type");
            }

            head_ = source.head_;
            item_ = source.item_;
            if (head_) {
                data_ = (T *)(item_->data());
                /*auto count = head_->item_byte_count() / sizeof(T);
                for (auto alloc_ptr = data_; count--; alloc_ptr++)
                {
                    new (alloc_ptr) T(std::forward<Args>(args)...);
                }*/
            }
            alias_ = source.alias_;

            source.data_ = nullptr;
            source.head_ = nullptr;
            source.item_ = nullptr;
            source.alias_ = false;
        }

        Pointer(const Pointer<T> &copy) = delete;

        inline auto &operator=(Pointer<T> &&assign) noexcept {
            acquire(std::move(assign));
            return *this;
        }

        inline auto &operator=(Pointer<uint8_t> &&assign) noexcept {
            acquire(std::move(assign));
            return *this;
        }

        Pointer<T> &operator=(const Pointer<T> &assign) = delete;

        __host__ __device__ inline T *get() const noexcept {
            return data_;
        }

        [[nodiscard]] inline T *operator->() const noexcept {
            return data_;
        }

        [[nodiscard]] inline bool is_alias() const noexcept {
            return alias_;
        }

        // This is used to replace cudaFree(), for adding the memory of destroyed object to the pool
        inline void release() noexcept {
            if (head_) {
                if (!std::is_trivially_destructible<T>::value) {
                    // Manual destructor calls
                    auto count = head_->item_byte_count() / sizeof(T);
                    for (auto alloc_ptr = data_; count--; alloc_ptr++) {
                        alloc_ptr->~T();
                    }
                }

                // Return the memory to pool
                head_->add(item_);
            }
            else if (data_ && !alias_) {
                // Free the memory
                cudaFree(data_);
            }

            data_ = nullptr;
            head_ = nullptr;
            item_ = nullptr;
            alias_ = false;
        }

        void acquire(Pointer<T> &other) noexcept {
            if (this == &other) {
                return;
            }

            release();

            data_ = other.data_;
            head_ = other.head_;
            item_ = other.item_;
            alias_ = other.alias_;
            other.data_ = nullptr;
            other.head_ = nullptr;
            other.item_ = nullptr;
            other.alias_ = false;
        }

        inline void acquire(Pointer<T> &&other) noexcept {
            acquire(other);
        }

        void acquire(Pointer<uint8_t> &other) {
            // Cannot acquire a non-pool pointer of different type
            if (!other.head_ && other.data_) {
                throw std::invalid_argument("cannot acquire a non-pool pointer of different type");
            }

            release();

            head_ = other.head_;
            item_ = other.item_;
            if (head_) {
                data_ = reinterpret_cast<T *>(item_->data());
                if (!std::is_trivially_constructible<T>::value) {
                    auto count = head_->item_byte_count() / sizeof(T);
                    for (auto alloc_ptr = data_; count--; alloc_ptr++) {
                        new(alloc_ptr) T;
                    }
                }
            }
            alias_ = other.alias_;
            other.data_ = nullptr;
            other.head_ = nullptr;
            other.item_ = nullptr;
            other.alias_ = false;
        }

        inline void acquire(Pointer<uint8_t> &&other) {
            acquire(other);
        }

        ~Pointer() noexcept {
            release();
        }

        [[nodiscard]] inline static Pointer<T> Owning(T *pointer) noexcept {
            return {pointer, false};
        }

        [[nodiscard]] inline static auto Aliasing(T *pointer) noexcept -> Pointer<T> {
            return {pointer, true};
        }

    private:
        Pointer(T *pointer, bool alias) noexcept: data_(pointer), alias_(alias) {
        }

        explicit Pointer(class MemoryPoolHead *head) {
            if (!head) {
                throw std::invalid_argument("head cannot be null");
            }
            head_ = head;
            item_ = head->get();
            data_ = reinterpret_cast<T *>(item_->data());
            if (!std::is_trivially_constructible<T>::value) {
                auto count = head_->item_byte_count() / sizeof(T);
                for (auto alloc_ptr = data_; count--; alloc_ptr++) {
                    new(alloc_ptr) T;
                }
            }
        }

        template<typename... Args>
        explicit Pointer(class MemoryPoolHead *head, Args &&... args) {
            if (!head) {
                throw std::invalid_argument("head cannot be null");
            }
            head_ = head;
            item_ = head->get();
            data_ = reinterpret_cast<T *>(item_->data());
            auto count = head_->item_byte_count() / sizeof(T);
            for (auto alloc_ptr = data_; count--; alloc_ptr++) {
                new(alloc_ptr) T(std::forward<Args>(args)...);
            }
        }

        T *data_ = nullptr;

        MemoryPoolHead *head_ = nullptr;

        MemoryPoolItem *item_ = nullptr;

        bool alias_ = false;
    };

    /**
    Returns a MemoryPoolHandle pointing to the global memory pool.
    */
    [[nodiscard]] inline static std::shared_ptr<util::MemoryPoolST> global_pool() noexcept {
        return global_variables::global_memory_pool;
    }

    [[nodiscard]] inline static std::shared_ptr<util::MemoryPoolST> New(bool clear_on_destruction = false) {
        return std::make_shared<util::MemoryPoolST>(clear_on_destruction);
    }

    // Allocate the memory
    template<typename T_>
    [[nodiscard]] inline static Pointer<T_> allocate(const std::shared_ptr<MemoryPool> &pool, size_t count) {
        //printf("pool head number %ld\n", pool->pool_count());
        return Pointer<T_>(pool->get_for_byte_count(count * sizeof(T_)));
    }
}
