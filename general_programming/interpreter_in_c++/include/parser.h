#ifndef MONKEY_PARSER_H_
#define MONKEY_PARSER_H_

#include <string>
#include <vector>
#include "token.h"
#include "lexer.h"
#include "ast.h"


namespace parser {
    typedef struct Parser {
        lexer::Lexer l;
        token::Token curToken;
        token::Token peekToken;
        void NextToken();
        ast::Program* ParseProgram();
        ast::Statement* ParseStatement();
        ast::LetStatement* ParseLetStatement();
        ast::ReturnStatement* ParseReturnStatement();
        bool ExpectPeek(token::TokenType t);
        bool CurTokenIs(token::TokenType t);
        bool PeekTokenIs(token::TokenType t);
        void PeekError(token::TokenType t);
        std::vector<std::string> errors;
        std::vector<std::string> Errors();
    } Parser;

    Parser New(lexer::Lexer &l);
};

#endif
