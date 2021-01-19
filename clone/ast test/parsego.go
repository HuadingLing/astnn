package main

import (
    "os"
    //"io/ioutil" 
    //"fmt"
    "go/parser"
    "go/token"
    "go/ast"
)

func main() {
    filepath := os.Args[2]
    //content ,err :=ioutil.ReadFile(filepath)
    fs := token.NewFileSet()
    tree, err := parser.ParseFile(fs, filepath, nil, 0)
    if err !=nil {
        panic(err) } 
    ast.Print(fs, tree)
    //fmt.Println(string(tree)) 
}
