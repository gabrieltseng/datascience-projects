#ifndef MONKEY_LEXER_H_
#define MONKEY_LEXER_H_

#include <string>

namespace lexer
{
    typedef struct {
        std::string input;
        int position;
        int readPosition;
        char ch;
    } Lexer;

    Lexer * New(std::string);
};

#endif
