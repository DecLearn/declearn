#!/bin/bash

# Copyright 2025 Inria (Institut National de Recherche en Informatique
# et Automatique)
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

: <<'DOC' # DOC keyword delimits the docstring of this script
Bash script to launch test pipelines for declearn.

This script comprises the following (types of) functions:

- Backend utilities to verbosely run (sets of) commands and report
  on their success/failure status:
    - run_command "<command> [arguments]*"
        Verbosely run a single string-defined command.
    - run_commands "context" ["<commmand> [arguments]+"]+
        Verbosely run a sequence of string-defined commands,
        and report on the number of failed ones afterwards.

- Functions that run (sets of) commands defining a category of tests:
    - lint_declearn_code
        Run the ruff and mypy commands on the source code.
    - lint_declearn_tests
        Run the ruff and mypy commands on the tests code.
    - run_declearn_tests [pytest_args]*
        Run unit and integrations tests, managing coverage computation
        and reporting, using pytest, flags to which may be provided.

- Functions that make up bits of the 'run_declearn_tests' one:
    - run_unit_tests [pytest_args]*
    - run_integration_tests [pytest_args]*

- The 'main' entry point, that enables calling any of the category of tests.

Instructions for developers/maintainers:
- For linters, you may add or edit the commands that are run, which are
  declared as an array of strings within the code of the 'lint_' functions.
- For tests, you may:
    - edit commands as part of any of the current backend functions
    - add (or remove) a backend function that runs one or multiple commands,
      and edit accordingly the array of commands under 'run_declearn_tests'
- If you want to add a new group of tests, write up a function for it, and
  edit 'main' to add an input argument case for triggering it.
DOC

run_command(){
    : '
    Run a given command, echoing it and being verbose about success/failure.

    Return:
        1 if the command failed, 0 if it was successful.
    '
    cmd="$*"
    name="\e[34m$(echo "$cmd" | cut -d' ' -f1)\e[0m"
    echo -e "\nRunning command: \e[34m$cmd\e[0m"
    if $cmd; then
        echo -e "$name command was \e[32msuccessful\e[0m"
        return 0
    else
        echo -e "$name command \e[31mfailed\e[0m"
        return 1
    fi
}


run_commands() {
    : '
    Verbosely run a set of commands and report whether all were a success.

    Syntax:
        run_commands "context string" ("command")+

    Return:
        The number of sub-commands that failed (0 if successful).
    '
    context=$1; shift; commands=( "$@" )
    n_cmd=${#commands[@]}
    declare -i failed
    echo "Running $context ($n_cmd commands)."
    for ((i = 0; i < n_cmd ; i++)); do
        run_command "${commands[i]}"
        # shellcheck disable=SC2181
        if [[ $? -ne 0 ]]; then failed+=1; fi
    done
    if [[ $failed -eq 0 ]]; then
        echo -e "\n\e[32mAll $n_cmd $context commands were successful.\e[0m"
    else
        echo -e "\n\e[31m$failed/$n_cmd $context commands failed.\e[0m"
    fi
    return $failed
}


lint_declearn_code() {
    : '
    Verbosely run linters on the declearn package source code.

    Return:
        The number of sub-commands that failed (0 if successful).
    '
    commands=(
        "ruff check declearn"
        "mypy --install-types --non-interactive declearn"
    )
    run_commands "declearn code static analysis" "${commands[@]}"
}


lint_declearn_tests() {
    : '
    Verbosely run linters on the declearn test suite source code.

    Return:
        The number of sub-commands that failed (0 if successful).
    '
    commands=(
        "ruff check test"
        "mypy --install-types --non-interactive --exclude=conftest.py test"
    )
    run_commands "declearn test code static analysis" "${commands[@]}"
}


run_declearn_tests() {
    : '
    Verbosely run the declearn test suite and export coverage results.

    Syntax:
        run_tests [optional_pytest_flags]

    Return:
        The number of sub-commands that failed (0 if successful).
    '
    # Remove any pre-existing coverage file.
    if [[ -f .coverage ]]; then rm .coverage; fi
    # Run the various sets of tests.
    commands=(
        "run_unit_tests $@"
        "run_integration_tests $@"
    )
    run_commands "declearn test suite" "${commands[@]}"
    status=$?
    # Display and export the cumulated coverage.
    coverage report --precision=2
    coverage xml
    # Return the success/failure status of tests.
    return $status
}


# shellcheck disable=SC2329
run_unit_tests() {
    : '
    Verbosely run the declearn unit tests (excluding integration ones).
    '
    echo "Running DecLearn unit tests."
    # shellcheck disable=SC2124
    command="pytest $@
        --cov=declearn --cov-append --cov-report=
        --ignore=test/functional/
        test
    "
    echo -e "\e[34m$command\e[0m"
    if $command; then return 0; else return 1; fi
}


# shellcheck disable=SC2329
run_integration_tests() {
    : '
    Verbosely run the declearn integration tests (skipping unit ones).
    '
    echo "Running DecLearn integration tests."
    # shellcheck disable=SC2124
    command="pytest $@
        --cov=declearn --cov-append --cov-report=
        test/functional/
    "
    echo -e "\e[34m$command\e[0m"
    if $command; then return 0; else return 1; fi
}


main() {
    if [[ $# -eq 0 ]]; then
        echo "Missing required positional argument."
        echo "Usage: {lint_code|lint_tests|run_tests} [PYTEST_FLAGS]*"
        exit 1
    fi
    action=$1; shift
    case $action in
        lc | lint_code)
            lint_declearn_code
            exit $?
            ;;
        lt | lint_tests)
            lint_declearn_tests
            exit $?
            ;;
        rt | run_tests)
            run_declearn_tests "$@"
            exit $?
            ;;
        *)
            echo "Usage: {lint_code|lint_tests|run_tests} [PYTEST_FLAGS]*"
            echo "Aliases for the action parameter values: {lc|lt|rt}."
            exit 1
            ;;
    esac
}


main "$@"
