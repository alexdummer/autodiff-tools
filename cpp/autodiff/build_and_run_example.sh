g++ neo-hooke-example.cpp -I../utils/ $(pkg-config --cflags eigen3 2>/dev/null) -o a.out && ./a.out
