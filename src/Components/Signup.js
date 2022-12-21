
import React from 'react'
import { Grid,Paper, Avatar, TextField, Button, Typography,Link } from '@material-ui/core'
import LockOutlinedIcon from '@material-ui/icons/LockOutlined';
import FormControlLabel from '@material-ui/core/FormControlLabel';
import Checkbox from '@material-ui/core/Checkbox';
import { useState } from 'react';
import axios from 'axios';
// pages is an array of strings representing different pages in the application
// formData is a state variable that holds the form data as an object
const pages = ["Adminsignin", "Adminactivities", 'Home'];
const [formData, setFormData] = useState({
  email: '',
  password: '',
  telephone: '',
});
// handleSubmit is called when the form is submitted
// it prevents the default form submission behavior and sends a POST request to the /api/users endpoint with the form data
const handleSubmit = (event) => {
  event.preventDefault();

  axios
    .post('/api/users', formData)
    .then((response) => {
      console.log(response.data);
      // redirect to the login page or show a success message
    })
    .catch((error) => {
      console.error(error);
      // show an error message
    });
};
// signup is a functional component that renders a form for users to sign up for an account
const signup=()=>{

    const paperStyle={padding :20,height:'70vh',width:280, margin:"20px auto"}
    const avatarStyle={backgroundColor:'#1bbd7e'}
    const btnstyle={margin:'8px 0'}
    return(
        <Grid>
            <Paper elevation={10} style={paperStyle}>
                <Grid align='center'>
                     <Avatar style={avatarStyle}><LockOutlinedIcon/></Avatar>
                    <h2>Sign up</h2>
                </Grid>
                <TextField
                  label="Email"
                  placeholder="Enter Email"
                  variant="outlined"
                  fullWidth
                  required
                  onChange={(event) =>
                    setFormData({ ...formData, email: event.target.value })
                  }
                />
                <TextField
                  label="Create Password"
                  placeholder="Enter password"
                  type="password"
                  variant="outlined"
                  fullWidth
                  required
                  onChange={(event) =>
                    setFormData({ ...formData, password: event.target.value })
                  }
                />
                <TextField
                  label="Confirm password"
                  placeholder="Confirm password"
                  type="password"
                  variant="outlined"
                  fullWidth
                  required
                  onChange={(event) =>
                    setFormData({ ...formData, passwordConfirm: event.target.value })
                  }
                />
               <TextField
                  label="Telephone"
                  placeholder="Enter telephone"
                  type="phone number"
                  variant="outlined"
                  fullWidth
                  required
                  onChange={(event) =>
                    setFormData({ ...formData, telephone: event.target.value })
                  }
                />
     

                <FormControlLabel
                    control={
                    <Checkbox
                        name="checkedB"
                        color="primary"
                    />
                    }
                    label="Remember me"
                 />
            <Link href="Adminsignin">   <Button type='submit' color='primary' variant="contained" style={btnstyle} fullWidth >Sign up</Button> </Link> 
                
            </Paper>
        </Grid>
    )
}

export default signup;
