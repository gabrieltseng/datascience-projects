#include "token.h"
#include "lexer.h"
#include "repl.h"
#include <string>
#include <iostream>


std::istream& getline_with_prompt(std::istream& in, std::string& line) {
    std::cout << repl::PROMPT;
    return std::getline(in, line);
};


namespace repl {
    void Start(std::istream& in, std::ostream& out) {
        for (std::string line; getline_with_prompt(in, line);) {
            lexer::Lexer l = lexer::New(line);
                token::Token tok = l.nextToken();
                while (tok.type != token::ENDOF) {
                    std::cout << "{" << tok.type << "," << tok.literal << "} \n";
                    tok = l.nextToken();
                }
        };
    };
};
