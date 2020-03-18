#include<string>
#include "lexer.h"
#include "token.h"


bool isLetter(char ch) {
    return std::isalpha(static_cast<unsigned char>(ch));
};

bool isDigit(char ch) {
    return std::isdigit(static_cast<unsigned char>(ch));
}


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

    char lexer::Lexer::peekChar() {
        if (readPosition >= input.length()) {
            return '\0';  // signifies EOF
        }
        else {
            return input[readPosition];
        }
    }

    std::string lexer::Lexer::readIdentifier() {
        int start_position = position;
        while (isLetter(ch)) {
            readChar();
        };
        return input.substr(start_position, position - start_position);
    };

    std::string lexer::Lexer::readNumber() {
        int start_position = position;
        while (isDigit(ch)) {
            readChar();
        }
        return input.substr(start_position, position - start_position);
    }

    void lexer::Lexer::skipWhitespace() {
        while (std::isspace(static_cast<unsigned char>(ch))) {
            readChar();
        };
    }

    token::Token lexer::Lexer::nextToken() {

            token::Token tok;

            skipWhitespace();

            // the switch used in the book behaved wierdly here
            if (ch == '=') {
                if (peekChar() == '=') {
                    char old_char = ch;
                    readChar();
                    std::string literal;
                    literal += old_char;
                    literal += ch;
                    tok.type = token::EQ;
                    tok.literal = literal;
                } else {
                    tok = newToken(token::ASSIGN, ch);
                }
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
            else if (ch == '*') {
                tok = newToken(token::ASTERISK, ch);
            }
            else if (ch == '/') {
                tok = newToken(token::SLASH, ch);
            }
            else if (ch == '-') {
                tok = newToken(token::MINUS, ch);
            }
            else if (ch == '>') {
                tok = newToken(token::GT, ch);
            }
            else if (ch == '<') {
                tok = newToken(token::LT, ch);
            }
            else if (ch == '!') {
                if (peekChar() == '=') {
                    char old_char = ch;
                    readChar();
                    std::string literal;
                    literal += old_char;
                    literal += ch;
                    tok.type = token::NOT_EQ;
                    tok.literal = literal;
                } else {
                    tok = newToken(token::BANG, ch);
                }
            }
            else if (ch == 0) {
                tok.literal = "";
                tok.type = token::ENDOF;
            }
            else {
                if (isLetter(ch)) {
                    tok.literal = readIdentifier();
                    tok.type = token::LookupIdent(tok.literal);
                    return tok;
                }
                else if (isDigit(ch)) {
                    tok.type = token::INT;
                    tok.literal = readNumber();
                    return tok;
                }
                else {
                    tok = newToken(token::ILLEGAL, ch);
                }
            }
            readChar();
            return tok;
        };

    lexer::Lexer& New(std::string &input){
        static lexer::Lexer l = {.input = input};
        l.readChar();
        return l;
    };
}
