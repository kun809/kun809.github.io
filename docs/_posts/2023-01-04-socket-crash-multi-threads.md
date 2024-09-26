---
layout: post
title: "Socket Crash in Multi-Threaded Environment"
categories: misc
---

## Issue Description

Our application is a web server load testing tool that performs a high volume of transactions with the web server in a short time. The crash occurs when using the application in a multi-threaded pattern. When the thread count reaches 50, the crash happens consistently.

Here is the relevant call stack from the crash dump:

```text
#0  0xf7f0b549 in __kernel_vsyscall ()
#1  0xf7ce7a37 in __pthread_kill_implementation (threadid=threadid@entry=3951033024, signo=signo@entry=6, no_tid=no_tid@entry=0) at ./nptl/pthread_kill.c:43
#2  0xf7ce7abf in __pthread_kill_internal (signo=6, threadid=3951033024) at ./nptl/pthread_kill.c:78
#3  0xf7c96685 in __GI_raise (sig=6) at ../sysdeps/posix/raise.c:26
#4  0xf7c7f451 in __GI_abort () at ./stdlib/abort.c:100
#5  0xf7cda3bc in __libc_message (action=<optimized out>, fmt=<optimized out>) at ../sysdeps/posix/libc_fatal.c:155
#6  0xf7cda41c in __GI___libc_fatal (message=0xeb7f9170 "Unexpected error 9 on netlink descriptor 397.\n") at ../sysdeps/posix/libc_fatal.c:164
#7  0xf7da1185 in __GI___netlink_assert_response (fd=397, result=-1) at ../sysdeps/unix/sysv/linux/netlink_assert_response.c:103
#8  0xf7da082e in make_request (pid=<optimized out>, fd=<optimized out>) at ../sysdeps/unix/sysv/linux/check_pf.c:171
#9  __check_pf (seen_ipv4=<optimized out>, seen_ipv6=<optimized out>, in6ai=<optimized out>, in6ailen=<optimized out>) at ../sysdeps/unix/sysv/linux/check_pf.c:329
#10 0xf7d5e301 in __GI_getaddrinfo (name=<optimized out>, service=<optimized out>, hints=<optimized out>, pai=<optimized out>) at ../sysdeps/posix/getaddrinfo.c:2289
...
```

## Cause of the Issue

In short, the crash is caused by socket mishandling between threads.

Our application uses sockets to establish connections with the server. When the server sends an HTTP header with the `connection: close` parameter, our application closes the socket. However, in some cases, the server may have already closed the socket before our application attempts to do so.

If this socket is then reallocated by the system and assigned to another thread—such as one handling DNS responses through the `getaddrinfo()` system call—this can cause a conflict. If the original thread tries to close the socket using its file descriptor (`fd`), it will cause a crash in the current thread. This happens because the `getaddrinfo()` function inside performs a validity check on the socket and aborts the program if it detects an invalid file descriptor ("bad file descriptor"), leaving our application no chance to handle the exception gracefully.

<center>
  <img src="/assets/socket-crash.png" alt="image" width="800" height="auto">
</center>