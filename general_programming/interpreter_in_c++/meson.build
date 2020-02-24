project(
    'cpp-monkey',
    'cpp',
    version : '1.0.0',
    license : 'MIT',
    default_options : ['cpp_std=c++11']) # compile for C++

# it will be referred from subdir projects
inc = include_directories('include')

# meson will try to find a meson.build file inside following directories
subdir('include')
subdir('src')
subdir('test')
