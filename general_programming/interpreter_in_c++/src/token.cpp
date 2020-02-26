#include "token.h"
#include <string>
#include <map>
#include <stdexcept>


namespace token {
    std::map<std::string, token::TokenType> keywords {
        {"fn", token::FUNCTION}, {"let", token::LET}
    };

    token::TokenType LookupIdent(std::string ident) {
        try {
            return keywords.at(ident);
        } catch (const std::out_of_range& oor) {
            return token::IDENT;
        };
    };
}
