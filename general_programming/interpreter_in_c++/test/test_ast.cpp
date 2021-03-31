#include "doctest.h"
#include "ast.h"
#include "token.h"
#include <vector>


TEST_CASE("Test String")
{
    ast::Program program = ast::Program();

    // manually populate the statement
    ast::LetStatement LetStatement;
    token::Token LetToken = {.type=token::LET, .literal="let"};
    token::Token MyVarToken = {.type=token::IDENT, .literal="myVar"};
    token::Token AnotherVarToken = {.type=token::IDENT, .literal="anotherVar"};

    ast::Identifier MyIdentifier;
    MyIdentifier.Token = MyVarToken;
    MyIdentifier.Value = "myVar";

    ast::Identifier AnotherIdentifier;
    AnotherIdentifier.Token = AnotherVarToken;
    AnotherIdentifier.Value = "anotherVar";

    LetStatement.Token = LetToken;
    LetStatement.Name = MyIdentifier;
    LetStatement.Value = &AnotherIdentifier;
    program.Statements.push_back(&LetStatement);

    CHECK(program.String() == "let myVar = anotherVar;");
};
