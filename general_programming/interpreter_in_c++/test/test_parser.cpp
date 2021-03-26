#include "doctest.h"
#include "token.h"
#include "lexer.h"
#include "parser.h"
#include "ast.h"
#include <string>
#include <vector>


bool testStatement(ast::Statement *s, std::string name) {
    if (s->TokenLiteral() != "let") {
        return false;
    }
    ast::LetStatement *derived = dynamic_cast<ast::LetStatement*>(s);
    if (derived) {
        if (derived->Name.Value != name) {
            return false;
        }
        else {
            return true;
        }
    }
    return false;
};


TEST_CASE("Test Let Statement")
{
    std::string input =
        "let x = 5; \n"
        "let y = 10; \n"
        "let foobar = 838383; \n";

    lexer::Lexer l = lexer::New(input);
    parser::Parser p = parser::New(l);
    ast::Program *program = p.ParseProgram();

    CHECK(program);  // ensure the program pointer is initialized
    if (program) {
        CHECK(program->Statements.size() == 3);  // the program should have 3 statements

        std::vector<std::string> ExpectedIdentifers{"x", "y", "foobar"};

        for (int testCase{ 0 }; testCase < ExpectedIdentifers.size(); testCase++) {
            ast::Statement *statement = program->Statements[testCase];
            CHECK(testStatement(statement, ExpectedIdentifers[testCase]) == true);
        };
    }
};
