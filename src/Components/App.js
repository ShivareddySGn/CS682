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

// List of pages that can be accessed by the admin
const pages = ["Adminsignin", "Adminactivities","signup"];

// The main component for the routing system
const App = () => {
    // Return the JSX for the component
    return ( 
    // Wrapper div for the component
    <div className="App">
        {/* Routes component for defining the application routes */}
        <Routes>
            {/* Route component for the adminsignin page */}
          <Route path="/" element={<Adminsignin />}/>
            {/* Route component for the admin Home page */}
            <Route path="/Home" element={<Home />}/>
            {/* Route component for the sign up page */}
            <Route path="/Signup" element={<Signup />}/>
            {/* Route component for the admin dashboard page */}
            <Route path="/Adminactivities" element={<Adminactivities />}/>
        </Routes>
    </div>);
}

// Export the component as the default export
export default App;  
