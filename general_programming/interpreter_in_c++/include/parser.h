#ifndef MONKEY_PARSER_H_
#define MONKEY_PARSER_H_

#include <string>
#include <vector>
#include "token.h"
#include "lexer.h"
#include "ast.h"


namespace parser {
    typedef struct {
        lexer::Lexer l;
        token::Token curToken;
        token::Token peekToken;
        void NextToken();
        ast::Program* ParseProgram();
        ast::Statement* ParseStatement();
        ast::LetStatement* ParseLetStatement();
        bool ExpectPeek(token::TokenType t);
        bool CurTokenIs(token::TokenType t);
        bool PeekTokenIs(token::TokenType t);
    } Parser;

    Parser& New(lexer::Lexer &l);
};

#endif
