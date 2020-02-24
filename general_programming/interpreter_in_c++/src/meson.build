libmonkey = shared_library(
    'monkey', # library name
    ['token/token.cpp', 'lexer/lexer.cpp'], # source files to be compiled
    include_directories : inc, # previously declared include directories in root :code:`meson.build`
    install : true)

monkeygen = executable(
    'monkeygen',
    'main.cpp',
    include_directories : inc,
    link_with : libmonkey,
    install : true)
