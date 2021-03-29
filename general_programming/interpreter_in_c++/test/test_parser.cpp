#include "doctest.h"
#include "token.h"
#include "lexer.h"
#include "parser.h"
#include "ast.h"
#include <string>
#include <vector>
#include <iostream>


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

bool checkParserErrors(parser::Parser *p) {
    if (p->errors.size() == 0) {
        return false;
    }
    else {
        std::cout << "Parser has " << p->errors.size() << " errors \n";
        for (int i = 0; i < p->errors.size(); i++) {
            std::cout << "parser error: " << p->errors[i] << "\n";
        }
        return true;
    }
}


TEST_CASE("Test Let Statement")
{
    std::string input =
        "let x = 5; \n"
        "let y = 10; \n"
        "let foobar = 838383; \n";
    lexer::Lexer l = lexer::New(input);
    parser::Parser p = parser::New(l);
    ast::Program *program = p.ParseProgram();
    REQUIRE(program);  // ensure the program pointer is initialized
    if (program) {
        REQUIRE(!checkParserErrors(&p));
        REQUIRE(program->Statements.size() == 3);  // the program should have 3 statements

        std::vector<std::string> ExpectedIdentifers{"x", "y", "foobar"};
        for (int testCase{ 0 }; testCase < ExpectedIdentifers.size(); testCase++) {
            ast::Statement *statement = program->Statements[testCase];
            CHECK(testStatement(statement, ExpectedIdentifers[testCase]) == true);
        };
    };
};


TEST_CASE("Test Return Statement")
{
    std::string input =
        "return 5; \n"
        "return 10; \n"
        "return 838383; \n";

    lexer::Lexer l = lexer::New(input);
    parser::Parser p = parser::New(l);
    ast::Program *program = p.ParseProgram();
    REQUIRE(program);  // ensure the program pointer is initialized
    if (program) {
        REQUIRE(!checkParserErrors(&p));
        CHECK(program->Statements.size() == 3);  // the program should have 3 statements

        for (int testCase{ 0 }; testCase < program->Statements.size(); testCase++) {
            CHECK(program->Statements[testCase]->TokenLiteral() == "return");
        };
    }
};
