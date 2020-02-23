#include "lexer.h"

namespace lexer{

    lexer::Lexer * New(std::string input){
        lexer::Lexer l = {input: input};
        return &l;
    };

}