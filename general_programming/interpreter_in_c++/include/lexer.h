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
        char peekChar();
        token::Token nextToken();
        std::string readIdentifier();
        std::string readNumber();
        void skipWhitespace();
    } Lexer;

    Lexer& New(std::string &input);
};

#endif
