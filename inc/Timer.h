/*
 * CTimer.h
 *
 *  Created on: Feb 16, 2018
 *      Author: cui
 */

#ifndef TIMER_H_
#define TIMER_H_

#include <chrono>
#include <string>
class Timer {
public:
  void start() {
    std::chrono::system_clock::now();
    m_StartTime = std::chrono::system_clock::now();
    m_bRunning = true;
  }

  void stop() {
    m_EndTime = std::chrono::system_clock::now();
    m_bRunning = false;
  }

  double elapsedMilliseconds() {
    std::chrono::time_point<std::chrono::system_clock> endTime;

    if (m_bRunning) {
      endTime = std::chrono::system_clock::now();
    } else {
      endTime = m_EndTime;
    }

    return std::chrono::duration_cast<std::chrono::milliseconds>(endTime -
                                                                 m_StartTime)
        .count();
  }

  double elapsedSeconds() { return elapsedMilliseconds() / 1000.0; }

  double print(const string &task) {
    double time = elapsedMilliseconds();
    std::cout << task << " costed:" << time << " milliseconds" << std::endl;
    return time;
  }

private:
  std::chrono::time_point<std::chrono::system_clock> m_StartTime;
  std::chrono::time_point<std::chrono::system_clock> m_EndTime;
  bool m_bRunning = false;
};

#endif /* TIMER_H_ */
