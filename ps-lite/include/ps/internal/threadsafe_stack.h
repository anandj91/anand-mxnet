/**
 *  Copyright (c) 2015 by Contributors
 */
#ifndef PS_INTERNAL_THREADSAFE_QUEUE_H_
#define PS_INTERNAL_THREADSAFE_QUEUE_H_
#include <stack>
#include <mutex>
#include <condition_variable>
#include <memory>
#include "ps/base.h"
namespace ps {

/**
 * \brief thread-safe stack allowing push and waited pop
 */
template<typename T> class ThreadsafeStack {
 public:
  ThreadsafeStack() { }
  ~ThreadsafeStack() { }

  /**
   * \brief push an value into the end. threadsafe.
   * \param new_value the value
   */
  void Push(T new_value) {
    mu_.lock();
    stack_.push(std::move(new_value));
    mu_.unlock();
    cond_.notify_all();
  }

  /**
   * \brief wait until pop an element from the beginning, threadsafe
   * \param value the poped value
   */
  void WaitAndPop(T* value) {
    std::unique_lock<std::mutex> lk(mu_);
    cond_.wait(lk, [this]{return !stack_.empty();});
    *value = std::move(stack_.top());
    stack_.pop();
  }

 private:
  mutable std::mutex mu_;
  std::stack<T> stack_;
  std::condition_variable cond_;
};

}  // namespace ps

// bool TryPop(T& value) {
//   std::lock_guard<std::mutex> lk(mut);
//   if(data_stack.empty())
//     return false;
//   value=std::move(data_stack.front());
//   data_stack.pop();
//   return true;
// }
#endif  // PS_INTERNAL_THREADSAFE_QUEUE_H_
