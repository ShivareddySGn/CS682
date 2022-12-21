import React from "react";
import { Typography, AppBar, Card, CardActions, CardContent, CardMedia, CssBaseline, Grid, Toolbar, Container, Button  } from '@material-ui/core';   
import ApiTwoToneIcon from '@mui/icons-material/ApiTwoTone';
import { makeStyles } from "@material-ui/core/styles";
import { green } from "@material-ui/core/colors";
import { CheckBox, Home, Label, Pages } from "@material-ui/icons";
import Checkbox from '@material-ui/core/Checkbox';
import { Link } from "react-router-dom";

// List of pages that can be accessed by the admin
const pages = ["Adminsignin", "Adminactivities" ];

// Custom hook for styling the component
const useStyles = makeStyles ((theme)=>({
    container : {
        backgroundColor : theme.palette.background.paper,
        padding : theme.spacing(8, 0, 6)
    },
    AddmodelButton:{
        marginLeft:'500px',
        padding:'20px',
    },
    DeletemodelButton:{
        marginLeft:'500px',
        padding:'10px',
    }
}));

// The main component for the admin dashboard
const Adminactivities = () => {
    // Use the custom hook for styling
    const classes = useStyles();
    return (
        // The component JSX
        <> 
         {/* CssBaseline component resets default styles for the page */}
         <CssBaseline />
         {/* AppBar component for the page header */}
             <AppBar position="relative" >
                {/* Toolbar component for the page header content */}
                <Toolbar>
                    {/* Icon component for the page logo */}
                    <ApiTwoToneIcon />
                    {/* Typography component for the page title */}
                    <Typography variant="h6">
                        Framework to compare machine learning models
                    </Typography>
                </ Toolbar> 
            </AppBar>
            {/* Main section of the page */}
            <main>
            {/* Container component for the welcome message */}
            <div className={classes.container} >
                <Container maxWidth='sm'>
                {/* Typography component for the welcome message title */}
                <Typography variant="h2" align="center " color="textSecondary" paragraph >
                        Welcome Admin.
                    </Typography>
                    {/* Typography component for the welcome message body */}
                    <Typography variant="h5" align="justify" color="textSecondary" paragraph >
                        Here you can add or delete deeplearning models to the framework. 
                    </Typography>
                </Container>
            </div>
            {/* Container component for the add and delete buttons */}
            <Container>
                {/* Button component for adding a model */}
                <div className={classes.AddmodelButton}>
                    <Button size="large"  variant="contained" color="primary"> Add model</Button>
                </div>
                {/* Button component for deleting a model */}
                <div className={classes.DeletemodelButton}>
                    <Button size="large" variant="contained" color="primary">  Delete model</Button>
                </div>
            </Container>
            </main>
        </>
    );
}
export default Adminactivities;
   
