<script lang="ts" setup>
import type { RecommendationResult } from '~/models/result';

const myResult = ref<RecommendationResult>();

const models = ['Model-1', 'Model-2', 'Model-3']
const selected = ref(models[0])

console.log('myResult.fileName =', myResult.value?.fileName)

const modelNum = computed(() => {
  return models.indexOf(selected.value);
});
watch(modelNum, (newNum) => {
  console.log('Selected model index:', newNum);
});

</script>

<template>
    <div class="w-full flex flex-col items-center justify-center">
        <div class="w-full md:w-3/4 lg:w-1/2 xl:w-1/3 p-4" v-if="!myResult">
            <h2 class="text-2xl my-4 font-semibold text-center">Explainable Course Recommendation System</h2>
            <p class="mb-4 text-left">
                To get a recommendation for a career role and courses to reach that career role, think of the courses you have already taken.
                <br>
                Group the courses you liked, courses you are neutral about, and courses you disliked. Write them by separating them with commas.
                <br>
                You do not need to write only the name of the course, you can write subjects, concepts you learned in courses or you already knew. (e.g. Java, Python, HTML, CSS, Spring Boot, Linux etc.)
                <br>
                Additionally, write subjects, concepts, or courses you are curious about.
            </p>
            <InputForm @submited="myResult = $event" />
        </div>

        <div class="w-full md:w-3/4 lg:w-1/2 xl:w-1/3 space-y-4 p-4" v-else>
            <h2 class="text-2xl my-4 font-semibold text-center">Recommendation Results</h2>
            <USelectMenu v-model="selected" :options="models" />

            <div v-if="selected === 'Model-1' || selected === 'Model-2' || selected === 'Model-3'">
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
                        <tr v-for="row in myResult.recommendations[modelNum].roles" :key="row.role + modelNum" class="border-2 border-slate-400">
                            <td colspan="4" class="px-4 py-2 bg-slate-100">
                                <div class="text-xl font-semibold">{{ row.role }}</div>
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
</template>