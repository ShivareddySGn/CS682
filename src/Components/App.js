import React from "react";
import { Typography, AppBar, Card, CardActions, CardContent, CardMedia, CssBaseline, Grid, Toolbar, Container, Button  } from '@material-ui/core';   
import ApiTwoToneIcon from '@mui/icons-material/ApiTwoTone';
import { makeStyles } from "@material-ui/core/styles";
import { green } from "@material-ui/core/colors";
import { CheckBox, Label, Pages } from "@material-ui/icons";
import { Link } from "react-router-dom";
import Home from "./Home"; 
import Adminsignin from "./Adminsignin";
import Signup from "./Signup";
import Adminactivities from "./Adminactivities";
import { Route, Routes } from 'react-router-dom';
const pages = ["Adminsignin", "Adminactivities","signup"];

const App = () => {
    
    return ( 
    <div className="App">
    <Routes>
      <Route path="/" element={<Home />}/>
      <Route path="/Adminsignin" element={<Adminsignin />}/>
      <Route path="/Signup" element={<Signup />}/>
      <Route path="/Adminactivities" element={<Adminactivities />}/>
    </Routes>
  </div>);
        
         
    

}
export default App;  

