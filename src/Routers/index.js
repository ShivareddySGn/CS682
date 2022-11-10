import React from "react";
import ReactDOM from "react-dom";
import App from "./App";
import { BrowserRouter  } from "react-router-dom";
// import Adminsignin from "./Components/Adminsignin"
ReactDOM.render(<BrowserRouter> <App /> </BrowserRouter> , document.getElementById('root')); 
// ReactDOM.render(<Adminsignin />, document.getElementById('root'));
