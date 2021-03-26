#include "lexer.h"
#include "parser.h"
#include "ast.h"
#include "token.h"


namespace parser {

    void parser::Parser::NextToken() {
        curToken = peekToken;
        peekToken = l.nextToken();
    };

    Parser& New (lexer::Lexer &l) {
        static Parser p = {.l = l};

        // Read two tokens, so curToken and peekToken are both set
        p.NextToken();
        p.NextToken();

        return p;
    };

    ast::LetStatement* parser::Parser::ParseLetStatement() {
        ast::LetStatement *s = new ast::LetStatement();
        s->Token = curToken;

        if (!ExpectPeek(token::IDENT)) {
            return nullptr;
        };

        ast::Identifier i = ast::Identifier();
        i.Token = curToken;
        i.Value = curToken.literal;
        s->Name = i;

        if (!ExpectPeek(token::ASSIGN)) {
            return nullptr;
        };

        // TODO - we will skip expressions until
        // we encounter a semicolon
        while (!CurTokenIs(token::SEMICOLON)) {
            NextToken();
        };

        return s;
    };

    ast::Statement* parser::Parser::ParseStatement() {
        ast::Statement *s = nullptr;
        if (curToken.type == token::LET) {
            s = ParseLetStatement();
        }
        return s;
    };

    ast::Program* parser::Parser::ParseProgram() {
        ast::Program *program = new ast::Program();
        while (curToken.type != token::ENDOF) {
            ast::Statement *s = ParseStatement();
            if (s) {
                program->Statements.push_back(s);
            };
            NextToken();
        };
        return program;
    };

    bool parser::Parser::ExpectPeek(token::TokenType t) {
        if (PeekTokenIs(t)) {
            NextToken();
            return true;
        }
        else {
            return false;
        };
    };

    bool parser::Parser::PeekTokenIs(token::TokenType t) {
        return peekToken.type == t;
    };

    bool parser::Parser::CurTokenIs(token::TokenType t) {
        return curToken.type == t;
    }
};
