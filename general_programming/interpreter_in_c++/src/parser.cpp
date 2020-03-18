#include "parser.h"
#include "lexer.h"


namespace parser {
    void parser::Parser::NextToken() {
        curToken = peekToken;
        peekToken = l.nextToken();
    }

    Parser& New(lexer::Lexer &l) {
        static Parser p = {.l = l};

        // Read two tokens, so curToken and peekToken are both set
        p.NextToken();
        p.NextToken();

        return p;
    }
}
