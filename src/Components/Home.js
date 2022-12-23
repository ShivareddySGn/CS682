import React from "react";
import { Typography, AppBar, Card, CardActions, CardContent, CardMedia, CssBaseline, Grid, Toolbar, Container, Button  } from '@material-ui/core';   
import ApiTwoToneIcon from '@mui/icons-material/ApiTwoTone';
import { makeStyles } from "@material-ui/core/styles";
import { green } from "@material-ui/core/colors";
import { CheckBox, Home, Label, Pages } from "@material-ui/icons";
import Checkbox from '@material-ui/core/Checkbox';
import { Link } from "react-router-dom";
import cardimage1 from '/Users/shivareddy/Desktop/material_ui/src/Images/bg1.png';
import cardimage2 from '/Users/shivareddy/Desktop/material_ui/src/Images/LSTM.png';
import FormControlLabel from '@material-ui/core/FormControlLabel';
import {useRef} from 'react';

// pages is an array of strings representing different pages in the application
const pages = ["Adminsignin","Adminactivities"];

// useStyles is a hook that returns the styles object for the component
const useStyles = makeStyles ((theme)=>({
  // styles for the main container
  container: {
    backgroundColor: theme.palette.background.paper,
    padding: theme.spacing(8, 0, 6),
  },
  // styles for the login button
  AdminButton: {
     marginLeft: 800,
  },
  // styles for the card grid container
  cardGrid: {
    padding: '20px',
  },
  // styles for the card
  card: {
    height: '100%',
    display: 'flex',
    flexDirection: 'column',
  },
  // styles for the card content
  cardContent: {
    flexGrow: '1',
  },
  // styles for the card media (image)
  cardMedia: {
    paddingTop: '56.26%', //16:9
  },
  // styles for the browse files button
  BrowsefilesButton: {
    marginLeft: '800px',
    padding: '20px',
  },
  // styles for the dataset button
  dataset: {
    marginLeft: '800px',
    padding: '10px',
  },
}));

const Homepage = () => {
  // classes is an object with the styles for the component
  const classes = useStyles();

  // ref is a reference to the component created with the useRef hook
  const ref = useRef();

  return (
    <> 
      {/* CssBaseline resets default styles for the page */}
      <CssBaseline />
      {/* AppBar displays a navbar at the top of the page */}
      <AppBar position="relative">
        <Toolbar>
          {/* ApiTwoToneIcon is a custom icon component */}
          <ApiTwoToneIcon />
          <Typography variant="h6">
            Framework to compare machine learning models
          

                    </Typography>
                    <div className={classes.AdminButton}>
                    <Grid  containerspacing= {2} justify="flex-end">
                        <Grid item>
                            <Button variant="contained"  
                              component = {Link}
                              to={
                                'Adminsignin'
                             }
                              color="primary" >
                                Login
                            </Button>
                        </Grid>

                    </Grid>
                    </div> 
                </ Toolbar> 
            </AppBar>
        <main>
            <div className={classes.container} >
                <Container maxWidth='sm'>
                <Typography variant="h2" align="center " color="textSecondary" paragraph >
                        Why compare?
                    </Typography>
                    <Typography variant="h5" align="justify" color="textSecondary" paragraph >
                        Chossing the right model for the problem saves time and resources for any project. Trade-off between different models in terms of time, accuracy on your dataset is neccesary to know to get deep insights on making right decisions before spending huge resporces on such project.  
                    </Typography>
                </Container>
            </div>
            <Container className={classes.cardGrid } maxWidth='md'>
                <Grid container spacing={4}>
                    <Grid item sm={6}>
                        <Card className={classes.card}>
                            <CardMedia className={classes.cardMedia} image={cardimage1} title = "Gated Recurring unit" />
                                <CardContent className={classes.cardContent}>
                                    
                                    <Typography gutterBottom variant="h5">
                                    Gated Recurrent unit 
                                    </Typography>
                                    <Typography>
                                    The GRU is like a long short-term memory (LSTM) with a forget gate, but has fewer parameters than LSTM, as it lacks an output gate. GRU's performance on certain tasks of polyphonic music modeling, speech signal modeling and natural language processing was found to be similar to that of LSTM. GRUs have been shown to exhibit better performance on certain smaller and less frequent datasets.
                                    </Typography>
                                </CardContent>
                                <CardActions>
                                    <Button size="small" color="primary">
                                        Learn more
                                    </Button>
                                    <FormControlLabel
                                    control={
                                    <Checkbox
                                        name="checkedB"
                                        color="primary"
                                    />
                                    }
                                    label="Select"
                                />
                                </CardActions>
                                
                            
                        </Card>
                    </Grid>
                    <Grid item sm={6}>
                        <Card className={classes.card}>
                            <CardMedia className={classes.cardMedia} image = {cardimage2} title = "Temporal Convolutional Networks" />
                                <CardContent className={classes.cardContent}>
                                    <Typography gutterBottom variant="h5">
                                     Long short-term memory
                                    </Typography>
                                    <Typography>
                                     In deep learning, a LSTM is a class of artificial neural network, most commonly applied to analyze visual imagery.They have applications in image and video recognition, recommender systems, image classification, image segmentation, medical image analysis, natural language processing, brain–computer interfaces, and financial time series.

                                    </Typography>
                                </CardContent>
                                <CardActions>
                                    <Button size="small" color="primary" padding='30px'>
                                        Learn more
                                    </Button>       
                                    <FormControlLabel
                                    control={
                                    <Checkbox
                                        name="checkedB"
                                        color="primary"
                                    />
                                    }
                                    label="Select"
                                />
                                    
                                </CardActions>
                                
                            
                        </Card>
                    </Grid>
                </Grid>

            </Container>
            <Container>
                <div className={classes.BrowsefilesButton}>
                <input
                    type="file"
                    accept="image/*"
                    style={{ display: 'none' }}
                    id="contained-button-file"
                />
                    <label htmlFor="contained-button-file">
                    <Button variant="contained" color="primary" component="span" size="large" > Browse files</Button>
                    </label>
                    
                </div>
                <div className={classes.datset}>
                    <Button size="large" variant="contained" color="primary">  Upload dataset</Button>
                </div>
            </Container>
        </main>
        </>    
    );

}
// Export the component as the default export
export default Homepage;  

