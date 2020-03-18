#include "ast.h"
#include "lexer.h"
#include "token.h"


namespace parser {
    typedef struct Parser {
        lexer::Lexer l;

        token::Token curToken;
        token::Token peekToken;

        void NextToken();
        ast::Program& ParseProgram();
    };

    Parser& New(lexer::Lexer &l);
}
