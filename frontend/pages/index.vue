<script lang="ts" setup>
import type { RecommendationResult } from '~/models/result';

definePageMeta({
    colorMode: 'light',
})

const myResult = ref<RecommendationResult>();

const isLoading = ref(false);

const models = ['Model-1', 'Model-2', 'Model-3', 'Model-4', 'Model-5']
const selected = ref(models[0])

console.log('myResult.fileName =', myResult.value?.fileName)

const modelNum = computed(() => {
  return models.indexOf(selected.value);
});
watch(modelNum, (newNum) => {
  console.log('Selected model index:', newNum);
});

const filteredRoles = computed(() => {
      const recommendationResult = myResult.value;
      if (!recommendationResult) {
        return [];
      }
      const recommendation = recommendationResult.recommendations[modelNum.value];

      if (!recommendation) {
        return [];
      }
      const roles = recommendation.roles;

      if (!roles) {
        return [];
      }

      return roles;
});

const courses = [
    // 0. Programming Languages
    ["Python", "Java", "C++", "C", "C#", "JavaScript", "Ruby", "PHP", "Swift", "Kotlin", "TypeScript", "Go", "Rust", "SQL", "R", "MATLAB", "Shell Scripting"],
    // 1. Web Development
    ["HTML", "CSS", "JavaScript", "React", "Angular", "Vue.js", "Node.js", "Express.js", "Next.js", "Bootstrap", "jQuery", "Django", "Flask", "Ruby on Rails", "ASP.NET", "Web Security", "RESTful APIs", "GraphQL"],
    // 2. Mobile Development
    ["iOS Development", "Android Development", "React Native", "Flutter", "Swift", "Kotlin", "Xamarin", "Mobile App Security"],
    // 3. Software Development
    ["Software Engineering", "Agile Methodologies", "DevOps", "Version Control (Git)", "Continuous Integration/Continuous Deployment (CI/CD)", "Test-Driven Development (TDD)", "Design Patterns", "Software Testing", "Microservices"],
    // 4. Data Structures & Algorithms
    ["Data Structures", "Algorithms", "Complexity Analysis", "Sorting and Searching", "Graph Theory", "Dynamic Programming", "Recursion", "Trees", "Hash Tables", "Heaps", "Linked Lists", "Stacks and Queues"],
    // 5. Databases
    ["SQL", "NoSQL", "MySQL", "PostgreSQL", "MongoDB", "Cassandra", "Oracle Database", "Redis", "Elasticsearch", "Database Design", "Data Warehousing", "Big Data"],
    // 6. Operating Systems
    ["Linux", "Windows", "macOS", "Unix", "Operating System Concepts", "Process Management", "Memory Management", "File Systems", "Shell Scripting"],
    // 7. Networks
    ["Computer Networks", "Network Protocols", "TCP/IP", "Network Security", "Wireless Networks", "Network Architecture", "Cloud Computing", "Network Administration"],
    // 8. Cybersecurity
    ["Cybersecurity Fundamentals", "Ethical Hacking", "Network Security", "Cryptography", "Information Security", "Penetration Testing", "Security Policies", "Risk Management", "Incident Response"],
    // 9. Cloud Computing
    ["AWS", "Azure", "Google Cloud Platform", "Cloud Architecture", "Cloud Security", "Kubernetes", "Docker", "Serverless Computing", "Cloud Storage", "Cloud Databases", "Cloud Services"],
    // 10. Artificial Intelligence & Machine Learning
    ["Artificial Intelligence", "Machine Learning", "Deep Learning", "Neural Networks", "Natural Language Processing", "Computer Vision", "Reinforcement Learning", "TensorFlow", "PyTorch", "Scikit-learn"],
    // 11. Data Science
    ["Data Analysis", "Data Visualization", "Data Mining", "Big Data", "Statistics", "Probability", "R", "Python for Data Science", "Pandas", "NumPy", "Matplotlib", "Seaborn", "Tableau", "Power BI"],
    // 12. Computer Graphics
    ["Computer Graphics", "Game Development", "Unity", "Unreal Engine", "OpenGL", "DirectX", "3D Modeling", "Animation", "VR/AR Development"],
    // 13. Embedded Systems
    ["Embedded Systems", "IoT", "Microcontrollers", "Arduino", "Raspberry Pi", "Real-Time Operating Systems", "Embedded C", "Firmware Development", "Robotics"],
    // 14. Theoretical Computer Science
    ["Computational Theory", "Automata Theory", "Formal Languages", "Compiler Design", "Complexity Theory", "Cryptography", "Information Theory", "Discrete Mathematics"],
    // 15. Miscellaneous
    ["Functional Programming", "Parallel Programming", "Distributed Systems", "Blockchain", "Quantum Computing", "Bioinformatics", "Human-Computer Interaction", "Software Project Management", "IT Governance", "IT Service Management"]
];

</script>

<template>
    <div class="w-full container mx-auto my-10">
        <div v-if="isLoading" class="w-screen h-screen z-10 fixed flex items-center justify-center top-0 left-0 bg-gray-400 opacity-50">Your inputs submitted, it may take 1-2 minutes to get a response...</div>

        <div class="flex items-center justify-center"> 
            <div class="w-full md:w-3/4 lg:w-1/2 xl:w-1/3 p-4" v-if="!myResult">
                <h2 class="text-2xl my-4 font-semibold text-center">Explainable Course Recommendation System</h2>
                <p class="mb-4 text-left">
                    To receive recommendations for career paths and courses to achieve your goals, consider the courses you've already completed. 
                    <br>
                    Categorize them into courses you enjoyed, courses you are neutral about, and courses you disliked, separating them with commas.
                    <br>
                    Feel free to include not just course names, but also subjects and concepts you've learned, such as Java, Python, HTML, CSS, Spring Boot, or Linux.
                    <br>
                    Additionally, please list any subjects, concepts, or courses you're curious about.
                    <br>
                    <span class="font-semibold">Note:</span>
                    <span class="italic"> There is a cheatsheet at the bottom of the page containing sample courses, subjects, concepts.</span>
                </p>


                <InputForm @submited="myResult = $event" v-model:is-loading="isLoading"/>

                <div class="my-4">
                    <!-- <h3 class="text-sm font-semibold">Course Cheatsheet:</h3> -->
                    <div class="flex flex-wrap gap-1 text-xs">
                        <div v-for="course in courses[0]" :key="course" class="p-1 bg-red-100 rounded inline-block" :style="{ minWidth: 'fit-content' }">
                            {{ course }}
                        </div>
                        <div v-for="course in courses[1]" :key="course" class="p-1 bg-orange-100 rounded inline-block" :style="{ minWidth: 'fit-content' }">
                            {{ course }}
                        </div>
                        <div v-for="course in courses[2]" :key="course" class="p-1 bg-emerald-100 rounded inline-block" :style="{ minWidth: 'fit-content' }">
                            {{ course }}
                        </div>
                        <div v-for="course in courses[3]" :key="course" class="p-1 bg-gray-100 rounded inline-block" :style="{ minWidth: 'fit-content' }">
                            {{ course }}
                        </div>
                        <div v-for="course in courses[4]" :key="course" class="p-1 bg-lime-100 rounded inline-block" :style="{ minWidth: 'fit-content' }">
                            {{ course }}
                        </div>
                        <div v-for="course in courses[5]" :key="course" class="p-1 bg-green-100 rounded inline-block" :style="{ minWidth: 'fit-content' }">
                            {{ course }}
                        </div>
                        <div v-for="course in courses[6]" :key="course" class="p-1 bg-amber-100 rounded inline-block" :style="{ minWidth: 'fit-content' }">
                            {{ course }}
                        </div>
                        <div v-for="course in courses[7]" :key="course" class="p-1 bg-teal-100 rounded inline-block" :style="{ minWidth: 'fit-content' }">
                            {{ course }}
                        </div>
                        <div v-for="course in courses[8]" :key="course" class="p-1 bg-yellow-100 rounded inline-block" :style="{ minWidth: 'fit-content' }">
                            {{ course }}
                        </div>
                        <div v-for="course in courses[9]" :key="course" class="p-1 bg-sky-100 rounded inline-block" :style="{ minWidth: 'fit-content' }">
                            {{ course }}
                        </div>
                        <div v-for="course in courses[10]" :key="course" class="p-1 bg-cyan-100 rounded inline-block" :style="{ minWidth: 'fit-content' }">
                            {{ course }}
                        </div>
                        <div v-for="course in courses[11]" :key="course" class="p-1 bg-indigo-100 rounded inline-block" :style="{ minWidth: 'fit-content' }">
                            {{ course }}
                        </div>
                        <div v-for="course in courses[12]" :key="course" class="p-1 bg-rose-100 rounded inline-block" :style="{ minWidth: 'fit-content' }">
                            {{ course }}
                        </div>
                        <div v-for="course in courses[13]" :key="course" class="p-1 bg-purple-100 rounded inline-block" :style="{ minWidth: 'fit-content' }">
                            {{ course }}
                        </div>
                        <div v-for="course in courses[14]" :key="course" class="p-1 bg-blue-100 rounded inline-block" :style="{ minWidth: 'fit-content' }">
                            {{ course }}
                        </div>
                        <div v-for="course in courses[15]" :key="course" class="p-1 bg-pink-100 rounded inline-block" :style="{ minWidth: 'fit-content' }">
                            {{ course }}
                        </div>
                    </div>
                </div>
            </div>        
            <div class="w-full md:w-3/4 lg:w-1/2 xl:w-1/3 space-y-4 p-4" v-else>
            <h2 class="text-2xl my-4 font-semibold text-center">Recommendation Results</h2>
            <USelectMenu v-model="selected" :options="models" />

            <div v-if="selected === 'Model-1' || selected === 'Model-2' || selected === 'Model-3' || selected === 'Model-4' || selected === 'Model-5'">
                <table class="table-auto w-full border-collapse">
                    <!-- <thead>
                        <tr>
                            <th class="px-4 py-2">Role</th>
                            <th class="px-4 py-2">Explanation</th>
                            <th class="px-4 py-2">Courses</th>
                        </tr>
                    </thead> -->
                    <tbody>
                        <!-- <tr>
                            <td colspan="4" class="px-4 py-2 text-center font-semibold ">{{ myResult.recommendations[modelNum].model }}</td>
                        </tr> -->
                        <tr v-for="(row, index_role) in filteredRoles" :key="row.role + modelNum" class="border-2 border-slate-400">
                            <td colspan="4" class="px-4 py-2 bg-slate-100">
                                <div class="text-xl font-semibold">Career Role {{index_role + 1}} - {{ row.role }}</div>
                                <!-- <div class="mt-2"><strong>Score:</strong> {{ row.score }}</div> -->
                                <div class="mt-2">{{ row.explanation }}</div>
                                <div class="mt-2">
                                    <div v-for="(course, index) in row.courses" :key="course.course + + modelNum" class="mt-2 bg-blue-100">
                                        <div class="text-lg font-bold mt-4 text-blue-800">Course {{ index + 1 }}</div>
                                        <div><strong>Title:</strong> {{ course.course }}</div>
                                        <div><strong>Explanation:</strong> {{ course.explanation }}</div>
                                        <div><strong>Url:</strong> <a :href="course.url" target="_blank" rel="noopener noreferrer" class="text-blue-500 underline">{{ course.url }}</a></div>
                                    </div>
                                </div>
                            </td>
                        </tr>
                    </tbody>
                </table>
            </div>

            <p class="pt-4 mb-4 text-left italic text-xl">
                Please fill the
                <a class="underline text-blue-500" :href="'https://docs.google.com/forms/d/e/1FAIpQLSdydUOOM0CvVoQUb6L4oCcmFxNKwbSOFzcBXK-jFFSodeBsYw/viewform?entry.1655930766='+ myResult.fileName">
                    form
                </a>
                by considering the results of models.
            </p>

            <UButton type="button" @click="myResult = undefined">
                Try again
            </UButton>
        </div>
        </div>


    </div>
</template>