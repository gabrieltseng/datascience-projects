#ifndef MONKEY_TOKEN_H_
#define MONKEY_TOKEN_H_

#include <string>
#include <map>

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
const TokenType MINUS = "-";
const TokenType BANG = "!";
const TokenType ASTERISK = "*";
const TokenType SLASH = "/";

// comparisons
const TokenType LT = "<";
const TokenType GT = ">";
const TokenType EQ = "==";
const TokenType NOT_EQ = "!=";

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
const TokenType TRUE = "TRUE";
const TokenType FALSE = "FALSE";
const TokenType IF = "IF";
const TokenType ELSE = "ELSE";
const TokenType RETURN = "RETURN";

token::TokenType LookupIdent(std::string ident);

};

#endif 
