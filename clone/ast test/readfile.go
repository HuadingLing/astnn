package main

import (
    "os"
    "io/ioutil" 
    "fmt"
)

func main() {
    filepath := os.Args[2]
    content ,err :=ioutil.ReadFile(filepath) 
    if err !=nil {
        panic(err) } 
    fmt.Println(string(content)) 
}