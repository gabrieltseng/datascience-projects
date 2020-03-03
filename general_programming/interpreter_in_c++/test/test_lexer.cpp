#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN

#include "doctest.h"
#include "token.h"
#include "lexer.h"
#include <string>
#include <array>


TEST_CASE("Test Next Token")
{
    std::string input =
        "let five = 5; \n"
        "let ten = 10; \n \n"
        "let add = fn(x, y) { \n"
        "   x + y; \n"
        "}; \n"
        "let result = add(five, ten); \n"
        "!-/*5; \n"
        "5 < 10 > 5; \n"
        "if (5 < 10) {\n"
        "   return true; \n"
        "} else { \n"
        "   return false; \n"
        "} \n"
        "10 == 10; \n"
        "10 != 9;";

    struct TestCases {
        token::TokenType expected_type;
        std::string expected_literal;
    };

    TestCases test_cases[74] = {
        {token::LET, "let"},
        {token::IDENT, "five"},
        {token::ASSIGN, "="},
        {token::INT, "5"},
        {token::SEMICOLON, ";"},
        {token::LET, "let"},
        {token::IDENT, "ten"},
        {token::ASSIGN, "="},
        {token::INT, "10"},
        {token::SEMICOLON, ";"},
        {token::LET, "let"},
        {token::IDENT, "add"},
        {token::ASSIGN, "="},
        {token::FUNCTION, "fn"},
        {token::LPAREN, "("},
        {token::IDENT, "x"},
        {token::COMMA, ","},
        {token::IDENT, "y"},
        {token::RPAREN, ")"},
        {token::LBRACE, "{"},
        {token::IDENT, "x"},
        {token::PLUS, "+"},
        {token::IDENT, "y"},
        {token::SEMICOLON, ";"},
        {token::RBRACE, "}"},
        {token::SEMICOLON, ";"},
        {token::LET, "let"},
        {token::IDENT, "result"},
        {token::ASSIGN, "="},
        {token::IDENT, "add"},
        {token::LPAREN, "("},
        {token::IDENT, "five"},
        {token::COMMA, ","},
        {token::IDENT, "ten"},
        {token::RPAREN, ")"},
        {token::SEMICOLON, ";"},
        {token::BANG, "!"},
        {token::MINUS, "-"},
        {token::SLASH, "/"},
        {token::ASTERISK, "*"},
        {token::INT, "5"},
        {token::SEMICOLON, ";"},
        {token::INT, "5"},
        {token::LT, "<"},
        {token::INT, "10"},
        {token::GT, ">"},
        {token::INT, "5"},
        {token::SEMICOLON, ";"},
        {token::IF, "if"},
        {token::LPAREN, "("},
        {token::INT, "5"},
        {token::LT, "<"},
        {token::INT, "10"},
        {token::RPAREN, ")"},
        {token::LBRACE, "{"},
        {token::RETURN, "return"},
        {token::TRUE, "true"},
        {token::SEMICOLON, ";"},
        {token::RBRACE, "}"},
        {token::ELSE, "else"},
        {token::LBRACE, "{"},
        {token::RETURN, "return"},
        {token::FALSE, "false"},
        {token::SEMICOLON, ";"},
        {token::RBRACE, "}"},
        {token::INT, "10"},
        {token::EQ, "=="},
        {token::INT, "10"},
        {token::SEMICOLON, ";"},
        {token::INT, "10"},
        {token::NOT_EQ, "!="},
        {token::INT, "9"},
        {token::SEMICOLON, ";"},
        {token::ENDOF, ""},
    };
    lexer::Lexer l = lexer::New(input);

    for(const auto& test_case : test_cases) {
        token::Token tok = l.nextToken();

        CHECK(tok.literal == test_case.expected_literal);
        CHECK(tok.type == test_case.expected_type);
    };
};
