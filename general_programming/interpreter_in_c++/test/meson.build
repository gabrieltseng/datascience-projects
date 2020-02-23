testexe = executable(
    'testexe', # test executable name 
    'lexer/test_lexer.cpp', # tests source files to be compiled
    include_directories : inc,  # declared include directories in root :code:`meson.build`
    link_with : libmonkey) # link test executable with previously declared shared library :code:`libuuid`

# test execution 
test('Interpreter test', testexe)

# we can specify other test execution passing arguments or environment variables
# test('Uuid test with args and env', testexe, args : ['arg1', 'arg2'], env : ['FOO=bar'])
