#ifndef MONKEY_LEXER_H_
#define MONKEY_LEXER_H_

#include <string>
#include "token.h"

namespace lexer
{
    typedef struct {
        std::string input;
        int position;
        int readPosition;
        char ch;
        void readChar();
        token::Token nextToken();
    } Lexer;

    Lexer New(std::string);
};

#endif
