<!DOCTYPE html>
<html>
<head>
    <title>Healthy Meal Plan</title>
    <script>
    async function getMealPlan() {
    let calories = document.getElementById('calories').value;
    let mealCount = document.getElementById('meal_count').value;

    let preferences = {};
    ['vegan', 'vegetarian', 'keto', 'paleo', 'gluten_free', 'mediterranean'].forEach(id => {
        if (document.getElementById(id).checked) {
            preferences[id] = 1;
        }
    });

    let data = {
        target_calories: calories,
        meal_count: mealCount,
        preferences: preferences
    };

    try {
        const response = await fetch('http://localhost:5000/recommend', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify(data)
        });

        const result = await response.json();
        let output = document.getElementById('output');

        if (response.ok) {
            output.innerHTML = '<h3>Recommended Meal Plan</h3>';
            result.meal_plan.forEach(meal => {
                output.innerHTML += `<p><b>${meal.meal_name}</b>: ${meal.calories} kcal, Protein ${meal.protein}g, Fat ${meal.fat}g, Carbs ${meal.carbs}g</p>`;
            });
            output.innerHTML += `<p><b>Total Calories:</b> ${result.total_calories}</p>`;
        } else {
            output.innerHTML = `<p style="color:red;">${result.error}</p>`;
        }
    } catch (error) {
        document.getElementById('output').innerHTML = `<p style="color:red;">Gagal menghubungi server: ${error.message}</p>`;
        console.error("Fetch error:", error);
    }
}

    </script>
</head>
<body>
    <h1>Healthy Meal Plan Generator</h1>
    <form onsubmit="event.preventDefault(); getMealPlan();">
        <label>Target Calories per Day: <input type="number" id="calories" value="2000" required></label><br><br>
        <label>Number of Meals: <input type="number" id="meal_count" value="3" min="1" max="5" required></label><br><br>
        <label><input type="checkbox" id="vegan">Vegan</label><br>
        <label><input type="checkbox" id="vegetarian">Vegetarian</label><br>
        <label><input type="checkbox" id="keto">Keto</label><br>
        <label><input type="checkbox" id="paleo">Paleo</label><br>
        <label><input type="checkbox" id="gluten_free">Gluten Free</label><br>
        <label><input type="checkbox" id="mediterranean">Mediterranean</label><br><br>
        <button type="submit">Generate Meal Plan</button>
    </form>

    <div id="output"></div>
</body>
</html>
