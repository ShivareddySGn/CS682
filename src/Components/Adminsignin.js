import React from 'react'
import { Grid,Paper, Avatar, TextField, Button, Typography,Link } from '@material-ui/core'
import LockOutlinedIcon from '@material-ui/icons/LockOutlined';
import FormControlLabel from '@material-ui/core/FormControlLabel';
import Checkbox from '@material-ui/core/Checkbox';

// List of pages that can be accessed by the admin
const pages = ["Adminsignin", "Adminactivities", 'Home'];

// The main component for the admin login page
const Adminsignin=()=>{
    // Styles for the Paper and Avatar components
    const paperStyle={padding :20,height:'70vh',width:280, margin:"20px auto"}
    const avatarStyle={backgroundColor:'#1bbd7e'}
    // Style for the submit button
    const btnstyle={margin:'8px 0'}
    return(
        // The component JSX
        <Grid>
            {/* Paper component for the login form container */}
            <Paper elevation={10} style={paperStyle}>
                {/* Grid component for the login form header */}
                <Grid align='center'>
                     {/* Avatar component for the login form logo */}
                     <Avatar style={avatarStyle}><LockOutlinedIcon/></Avatar>
                    {/* Typography component for the login form title */}
                    <h2>Sign In</h2>
                </Grid>
                {/* TextField component for the username input */}
                <TextField label='Username' placeholder='Enter username' variant="outlined" fullWidth required/>
                {/* TextField component for the password input */}
                <TextField label='Password' placeholder='Enter password' type='password' variant="outlined" fullWidth required/>
                {/* FormControlLabel component for the remember me checkbox */}
                <FormControlLabel
                    control={
                    {/* Checkbox component for the remember me checkbox */}
                    <Checkbox
                        name="checkedB"
                        color="primary"
                    />
                    }
                    label="Remember me"
                 />
                {/* Button component for submitting the login form */}
            <Link href="Adminactivities">   <Button type='submit' color='primary' variant="contained" style={btnstyle} fullWidth >Sign in</Button> </Link> 
                {/* Typography component for the password reset link */}
                <Typography >
                     <Link href="#" >
                        Forgot password ?
                </Link>
                </Typography>
                {/* Typography component for the sign up link */}
                                <Typography > Do you have an account ?
                     <Link href="Signup" >
                        Sign Up 
                </Link>
                </Typography>
            </Paper>
        </Grid>
    )
}

// Export the component as the default export
export default Adminsignin;

