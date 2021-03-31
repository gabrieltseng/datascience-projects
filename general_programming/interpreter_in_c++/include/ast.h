#ifndef MONKEY_AST_H_
#define MONKEY_AST_H_

#include <string>
#include <vector>
#include "token.h"

namespace ast
{
    typedef struct Node {
        virtual std::string TokenLiteral() {};
        virtual std::string String() {};
        virtual ~Node() {};
    } Node;

    typedef struct Statement: Node {
        void StatementNode();
    } Statement;

    typedef struct Expression: Node {
        void ExpressionNode();
    } Expression;

    typedef struct Program: Node {
        std::vector<Statement *> Statements;
        virtual std::string TokenLiteral();
        virtual std::string String();
    } Program;

    typedef struct Identifier: Expression {
        token::Token Token; // IDENT token
        std::string Value;
        void ExpressionNode();
        virtual std::string TokenLiteral();
        virtual std::string String();
    } Identifier;

    typedef struct LetStatement: Statement {
        token::Token Token;
        Identifier Name;
        Expression *Value;
        void StatementNode();
        virtual std::string TokenLiteral();
        virtual std::string String();
    } LetStatement;

    typedef struct ReturnStatement: Statement {
        token::Token Token;
        Expression *ReturnValue;
        void StatementNode();
        virtual std::string TokenLiteral();
        virtual std::string String();
    } ReturnStatement;

    typedef struct ExpressionStatement: Statement {
        token::Token Token;
        Expression *Expression;
        void StatementNode();
        virtual std::string TokenLiteral();
        virtual std::string String();
    } ExpressionStatement;
};

#endif
