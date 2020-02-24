#include<string>
#include "lexer.h"
#include "token.h" 


namespace lexer{

    token::Token newToken(token::TokenType tokenType, char ch) {
        std::string ch_str;

        ch_str += ch;
        token::Token token = {tokenType, ch_str};
        return token;
    };

    void lexer::Lexer::readChar() {
        if (readPosition >= input.length()) {
            ch = '\0';  // signifies EOF
        }
        else {
            ch = input[readPosition];
        }
        position = readPosition;
        readPosition += 1;
        };

    token::Token lexer::Lexer::nextToken() {
            
            token::Token tok;

            // the switch used in the book behaved wierdly here
            if (ch == '=') {
                tok = newToken(token::ASSIGN, ch);
            }
            else if (ch == '+') {
                tok = newToken(token::PLUS, ch);
            }
            else if (ch == ';') {
                tok = newToken(token::SEMICOLON, ch);
            }
            else if (ch == '(') {
                tok = newToken(token::LPAREN, ch);
            }
            else if (ch == ')') {
                tok = newToken(token::RPAREN, ch);
            }
            else if (ch == ',') {
                tok = newToken(token::COMMA, ch);
            }
            else if (ch == '{') {
                tok = newToken(token::LBRACE, ch);
            }
            else if (ch == '}') {
                tok = newToken(token::RBRACE, ch);
            }
            else if (ch == 0) {
                tok.literal = "";
                tok.type = token::ENDOF;
            }
            readChar();
            return tok;
        };

    lexer::Lexer New(std::string input){
        lexer::Lexer l = {input: input};
        l.readChar();
        return l;
    };
}
