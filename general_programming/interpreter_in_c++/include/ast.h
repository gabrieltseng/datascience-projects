#ifndef MONKEY_AST_H_
#define MONKEY_AST_H_

#include <string>
#include <vector>
#include "token.h"

namespace ast
{
    typedef struct Node {
        virtual std::string TokenLiteral() {};
        virtual ~Node() {};
    } Node;

    typedef struct Statement: Node {
        void StatementNode();
    } Statement;

    typedef struct Expression: Node {
        void ExpressionNode();
    } Expression;

    typedef struct Program {
        std::vector<Statement *> Statements;
        std::string TokenLiteral();
    } Program;

    typedef struct Identifier: Expression {
        token::Token Token; // IDENT token
        std::string Value;
        void ExpressionNode();
        virtual std::string TokenLiteral();
    } Identifier;

    typedef struct LetStatement: Statement {
        token::Token Token;
        Identifier Name;  // this should be a pointer
        void StatementNode();
        virtual std::string TokenLiteral();
    } LetStatement;

    typedef struct ReturnStatement: Statement {
        token::Token Token;
        Expression ReturnValue;
        void StatementNode();
        virtual std::string TokenLiteral();
    } ReturnStatement;
};

#endif
