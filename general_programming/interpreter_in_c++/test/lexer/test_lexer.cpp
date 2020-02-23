#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN

#include "doctest.h"
#include "token.h"
#include "lexer.h"
#include <string>
#include <array>


TEST_CASE("Test Nest Token")
{
    std::string input = "=+(){},;";

    struct TestCases {
        token::TokenType expected_type;
        std::string expected_literal;
    };

    TestCases test_cases[9] = {
        {token::ASSIGN, "="},
        {token::PLUS, "+"},
        {token::LPAREN, "("},
        {token::RPAREN, ")"},
        {token::LBRACE, "{"},
        {token::RBRACE, "}"},
        {token::COMMA, ","},
        {token::SEMICOLON, ";"},
        {token::ENDOF, "EOF"},
    };
    
    lexer::Lexer l = lexer::New(input);

    for(const auto& test_case : test_cases) {
        token::TokenType tok = l.NextToken();

        CHECK(tok.type == test_case.expected_type);
        CHECK(tok.literal == test_case.expected_literal);
    };
};
