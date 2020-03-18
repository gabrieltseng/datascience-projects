#include <string>
#include <vector>
#include "token.h"


namespace ast {
    typedef struct Node {
        std::string TokenLiteral();
    };

    typedef struct Statement: Node {
        void statementNode();
    };

    typedef struct Expression: Node {
        void expressionNode();
    };

    typedef struct Program: Node {
        std::vector<Statement> Statements;
        std::string TokenLiteral();
    };

    typedef struct Identifier: Expression {
        token::Token Token;  // the IDENT token
        std::string Value;
        void expressionNode();
        std::string TokenLiteral();
    };

    typedef struct LetStatement: Statement {
        token::Token Token;  // the LET token
        Identifier *Name;
        Expression Value;
        void statementNode();
        std::string TokenLiteral();
    };
};
