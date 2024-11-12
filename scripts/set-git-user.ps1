$username = "Jommel Limchu"
$email = "limchu002@gmail.com"

Set-Location ..
git config --global user.name $username
git config --global user.email $email

Write-Host "Git global username and email have been set to '$username' and '$email'"