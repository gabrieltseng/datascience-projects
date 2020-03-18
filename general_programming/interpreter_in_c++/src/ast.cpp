#include "ast.h"


namespace ast {
    std::string ast::Program::TokenLiteral() {
        if (Statements.size() > 0) {
            return Statements[0].TokenLiteral();
        }
    };

    std::string ast::Identifier::TokenLiteral() {
        return Token.literal;
    };

    std::string ast::LetStatement::TokenLiteral() {
        return Token.literal;
    };
}
