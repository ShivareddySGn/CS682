import React from "react";
import { Typography, AppBar, Card, CardActions, CardContent, CardMedia, CssBaseline, Grid, Toolbar, Container, Button  } from '@material-ui/core';   
import ApiTwoToneIcon from '@mui/icons-material/ApiTwoTone';
import { makeStyles } from "@material-ui/core/styles";
import { green } from "@material-ui/core/colors";
import { CheckBox, Home, Label, Pages } from "@material-ui/icons";
import Checkbox from '@material-ui/core/Checkbox';
import { Link } from "react-router-dom";
const pages = ["Adminsignin", "Adminactivities" ];
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
const Adminactivities = () => {
    const classes = useStyles();
    return (
        <> 
         <CssBaseline />
             <AppBar position="relative" >
                <Toolbar>
                    <ApiTwoToneIcon />
                    <Typography variant="h6">
                        Framework to compare machine learning models
                    </Typography>
                </ Toolbar> 
            </AppBar>
            <main>
            <div className={classes.container} >
                <Container maxWidth='sm'>
                <Typography variant="h2" align="center " color="textSecondary" paragraph >
                        Welcome Admin.
                    </Typography>
                    <Typography variant="h5" align="justify" color="textSecondary" paragraph >
                        Here you can add or delete deeplearning models to the framework. 
                    </Typography>
                </Container>
            </div>
            <Container>
                <div className={classes.AddmodelButton}>
                    <Button size="large"  variant="contained" color="primary"> Add model</Button>
                </div>
                <div className={classes.DeletemodelButton}>
                    <Button size="large" variant="contained" color="primary">  Delete model</Button>
                </div>
            </Container>
            </main>
        </>
    );
}
export default Adminactivities;
   