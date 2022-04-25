
(()=> {	modeSwitcher()})();
// let systemInitiatedDark = window.matchMedia("(prefers-color-scheme: dark)");
document.documentElement.setAttribute('data-theme', 'light')
sessionStorage.setItem('theme', 'light');
let theme = sessionStorage.getItem('theme');


function modeSwitcher() {
	let theme = sessionStorage.getItem('theme');
	if (theme === "dark") {
		document.documentElement.setAttribute('data-theme', 'light');
		sessionStorage.setItem('theme', 'light');
		// document.getElementById("theme-toggle").innerHTML = "Dark Mode";
	}	else if (theme === "light") {
		document.documentElement.setAttribute('data-theme', 'dark');
		sessionStorage.setItem('theme', 'dark');
		// document.getElementById("theme-toggle").innerHTML = "Light Mode";
	} else {
		document.documentElement.setAttribute('data-theme', 'dark');
		sessionStorage.setItem('theme', 'dark');
		// document.getElementById("theme-toggle").innerHTML = "Light Mode";
	}
}

// } else if (systemInitiatedDark.matches) {
// 	document.documentElement.setAttribute('data-theme', 'light');
// 	sessionStorage.setItem('theme', 'light');
	//let theme = sessionStorage.getItem('theme');
	//console.log("this was triggered");
	// document.getElementById("theme-toggle").innerHTML = "Dark Mode";