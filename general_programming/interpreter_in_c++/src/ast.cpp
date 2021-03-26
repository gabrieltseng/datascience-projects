#include<string>
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

    std::string ast::LetStatement::TokenLiteral() {
        return Token.literal;
    }

    std::string ast::Identifier::TokenLiteral() {
        return Token.literal;
    };
};
