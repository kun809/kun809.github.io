# `GLIBCXX_3.4.26` not found
## Introduction
I was upgrading gcc version from 5 to 8 for my Linux project. An error occurred when I compiled the program with gcc-9 and ran it on an Ubuntu18.04 machine.
```
System error:/usr/lib32/libstdc++.so.6:version `GLIBCXX_3.4.26' not found 
```
It seems a compatibility issue and I wrote a simple program named `test` to reproduce this problem:
```c++
#include <iostream>
#include <sstream>

using namespace std;

int main()
{
    cout << "hello world!" << endl;
    ostringstream s;
    return 0;
}
```
Compile with gcc-9 and run it on an Ubuntu18.04 machine(without any compiler installed):
```console
root@ubuntu:~$ ./test
./test: /usr/lib/x86_64-linux-gnu/libstdc++.so.6: version `GLIBCXX_3.4.26' not found (required by ./test)
root@ubuntu:~$ ldd ./test 
./test: /usr/lib/x86_64-linux-gnu/libstdc++.so.6: version `GLIBCXX_3.4.26' not found (required by ./test)
        linux-vdso.so.1 (0x00007ffc903b1000)
        libstdc++.so.6 => /usr/lib/x86_64-linux-gnu/libstdc++.so.6 (0x00007f74d4627000)
        libc.so.6 => /lib/x86_64-linux-gnu/libc.so.6 (0x00007f74d4236000)
        libm.so.6 => /lib/x86_64-linux-gnu/libm.so.6 (0x00007f74d3e98000)
        /lib64/ld-linux-x86-64.so.2 (0x00007f74d4bb2000)
        libgcc_s.so.1 => /lib/x86_64-linux-gnu/libgcc_s.so.1 (0x00007f74d3c80000)
```
It told me that `GLIBCXX_3.4.26` can not be found from the `libstdc++.so.6`.
## Reason & Solution
I googled it and the answer I got was I needed to install gcc-8 and update `libstdc++.so.6` on my Ubuntu machine. It works, but it can not be the final solution since I don't want to install gcc-8 each time when I run it on a new machine. So I need to find a better one.

I found that the ABI version(GLIBCXX_**) is different when the program is compiled by a different version of gcc. Let's do more tests with the `test` program:

Compile with gcc-9, and check the GLIBCXX:
```console
root@ubuntu:~$ g++ ./main.cpp -o test
root@ubuntu:~$ strings ./test | grep GLIBCXX
GLIBCXX_3.4.26
GLIBCXX_3.4
GLIBCXX_3.4.21
_ZNSt7__cxx1119basic_ostringstreamIcSt11char_traitsIcESaIcEED1Ev@@GLIBCXX_3.4.21
_ZSt4endlIcSt11char_traitsIcEERSt13basic_ostreamIT_T0_ES6_@@GLIBCXX_3.4
_ZStlsISt11char_traitsIcEERSt13basic_ostreamIcT_ES5_PKc@@GLIBCXX_3.4
_ZNSolsEPFRSoS_E@@GLIBCXX_3.4
_ZSt4cout@@GLIBCXX_3.4
_ZNSt7__cxx1119basic_ostringstreamIcSt11char_traitsIcESaIcEEC1Ev@@GLIBCXX_3.4.26
_ZNSt8ios_base4InitC1Ev@@GLIBCXX_3.4
_ZNSt8ios_base4InitD1Ev@@GLIBCXX_3.4
```
compile with gcc-8, and check the GLIBCXX again:
```console
root@ubuntu:~$ g++ ./main.cpp -o test
root@ubuntu:~$ strings ./test | grep GLIBCXX
GLIBCXX_3.4
GLIBCXX_3.4.21
_ZNSt7__cxx1119basic_ostringstreamIcSt11char_traitsIcESaIcEED1Ev@@GLIBCXX_3.4.21
_ZSt4endlIcSt11char_traitsIcEERSt13basic_ostreamIT_T0_ES6_@@GLIBCXX_3.4
_ZNSt7__cxx1119basic_ostringstreamIcSt11char_traitsIcESaIcEEC1ESt13_Ios_Openmode@@GLIBCXX_3.4.21
_ZStlsISt11char_traitsIcEERSt13basic_ostreamIcT_ES5_PKc@@GLIBCXX_3.4
_ZNSolsEPFRSoS_E@@GLIBCXX_3.4
_ZSt4cout@@GLIBCXX_3.4
_ZNSt8ios_base4InitC1Ev@@GLIBCXX_3.4
_ZNSt8ios_base4InitD1Ev@@GLIBCXX_3.4
```
They have some differences in GLIBCXX requirements: gcc-9 requires GLIBCXX_3.4.26 while gcc-8 doesn't. 
I referred to the [GNU page](https://gcc.gnu.org/onlinedocs/libstdc++/manual/abi.html) to find the reason for the differences. It gives a nutshell: *library API + compiler ABI = library ABI* to explain the rationale of library ABI.

GLIBCXX versions on my Ubuntu18.04
```console
root@ubuntu:~$ strings /usr/lib/x86_64-linux-gnu/libstdc++.so.6 | grep GLIBCXX
GLIBCXX_3.4
GLIBCXX_3.4.1
GLIBCXX_3.4.2
GLIBCXX_3.4.3
GLIBCXX_3.4.4
GLIBCXX_3.4.5
GLIBCXX_3.4.6
GLIBCXX_3.4.7
GLIBCXX_3.4.8
GLIBCXX_3.4.9
GLIBCXX_3.4.10
GLIBCXX_3.4.11
GLIBCXX_3.4.12
GLIBCXX_3.4.13
GLIBCXX_3.4.14
GLIBCXX_3.4.15
GLIBCXX_3.4.16
GLIBCXX_3.4.17
GLIBCXX_3.4.18
GLIBCXX_3.4.19
GLIBCXX_3.4.20
GLIBCXX_3.4.21
GLIBCXX_3.4.22
GLIBCXX_3.4.23
GLIBCXX_3.4.24
GLIBCXX_3.4.25
GLIBCXX_DEBUG_MESSAGE_LENGTH
```
It doesn't contain `GLIBCXX_3.4.26`. And I think the best solution for me is to use gcc-8 instead of gcc-9. 

And this error also reminds me that I have to choose a proper gcc version to be compatible with my Linux environment.