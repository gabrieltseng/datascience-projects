#include <string>
#include <sstream>
#include "ast.h"
#include "token.h"


namespace ast {
    std::string ast::Program::TokenLiteral() {
        if (Statements.size() > 0) {
            return Statements[0]->TokenLiteral();
        }
        else {
            return "";
        }
    };

    std::string ast::Program::String() {
        std::ostringstream stream;

        for (int statementIdx{ 0 }; statementIdx < Statements.size(); statementIdx++) {
            stream << Statements[statementIdx]->String();
        };
        return stream.str();
    }

    std::string ast::LetStatement::TokenLiteral() {
        return Token.literal;
    }

    std::string ast::LetStatement::String() {
        std::ostringstream stream;

        stream << TokenLiteral() << " " << Name.String() << " = ";
        if (Value) {
            stream << Value->String();
        }
        stream << ";";
        return stream.str();
    }

    std::string ast::ReturnStatement::TokenLiteral() {
        return Token.literal;
    };

    std::string ast::ReturnStatement::String() {
        std::ostringstream stream;
        stream << TokenLiteral() << " ";
        if (ReturnValue) {
            stream << ReturnValue->String();
        };
        stream << ";";
        return stream.str();
    }

    std::string ast::Identifier::TokenLiteral() {
        return Token.literal;
    };

    std::string ast::Identifier::String() {
        return Value;
    };

    std::string ast::ExpressionStatement::TokenLiteral() {
        return Token.literal;
    };

    std::string ast::ExpressionStatement::String() {
        if (Expression) {
            return Expression->String();
        };
        return "";
    };
};
