#include "token.h"
#include "lexer.h"
#include <string>
#include <iostream>

namespace repl {
    const std::string PROMPT = ">>";
    void Start(std::istream& in, std::ostream& out);
}
