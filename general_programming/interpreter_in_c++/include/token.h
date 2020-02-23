#ifndef MONKEY_TOKEN_H_
#define MONKEY_TOKEN_H_

#include <string>

namespace token
{
typedef std::string TokenType;

typedef struct {TokenType type; std::string literal;} Token;

const TokenType ILLEGAL = "ILLEGAL";
const TokenType ENDOF = "EOF";  // EOF is protected in C++

// Identifiers + literals
const TokenType IDENT = "IDENT"; // add, foobar, x, y, ...
const TokenType INT = "INT"; // 1343456

// Operators
const TokenType ASSIGN = "=";
const TokenType PLUS = "+";

// Delimiters
const TokenType COMMA = ",";
const TokenType SEMICOLON = ";";

const TokenType LPAREN = "(";
const TokenType RPAREN = ")";
const TokenType LBRACE = "{";
const TokenType RBRACE = "}";

// Keywords
const TokenType FUNCTION = "FUNCTION";
const TokenType LET = "LET";
};

#endif 
