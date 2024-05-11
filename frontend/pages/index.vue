<script lang="ts" setup>
import type { RecommendationResult } from '~/models/result';

const myResult = ref<RecommendationResult>();

const models = ['Model-1', 'Model-2', 'Model-3']
const selected = ref(models[0])

console.log('myResult.fileName =', myResult.value?.fileName)


</script>


<template>
    <div class=" w-full flex flex-col items-center justify-center">
        <div class="w-80" v-if="!myResult">
            <h2 class="text-2xl my-4  font-semibold">Explainable course recommendation system</h2>
            <p class="mb-4">To get a recommendation for career role and courses to reach that career role, think of the courses you have already took. <br>
                Group the courses you liked, courses you are neutral and courses you disliked. Write them by seperating by commas. <br>
                If you don't remember the full name of the course, you can write the main concepts you learned in that course. <br>
                Additionally write topics, concepts or courses you are curious about.
            </p>

            <InputForm @submited="myResult = $event"></InputForm>
 
        </div>
        <div class="w-80 space-y-4" v-else>

            <USelectMenu v-model="selected" :options="models" />

            <div class="w-80" v-if="selected === 'Model-1'">
                <h2 class="text-2xl my-4  font-semibold">Palm</h2>
                <table class="table">
                    <tbody>
                        <tr>{{ myResult.recommendations[0].model }}</tr>
                        <tr v-for="row in myResult.recommendations[0].roles" :key="row.role">
                            <tr> 
                                <td>Role</td>
                                <td>{{ row.role }}</td>  
                            </tr> 
                            <tr>
                                <td>Explanation</td>
                                <td>{{ row.explanation }}</td> 
                            </tr>
                            <tr>    
                                <td>Courses</td>
                                <tr v-for="course in row.courses">
                                    <tr>Title: {{ course.course }}</tr>
                                    <tr>Url: {{ course.url }}</tr>
                                    <tr>Explanation: {{ course.explanation }}</tr>
                                </tr>
                            </tr>                      
                        </tr>                    
                    </tbody>
                </table>
            </div>
            <div class="w-80" v-else-if="selected === 'Model-2'">
                <h2 class="text-2xl my-4  font-semibold">Voyage</h2>
                <table class="table">
                    <tbody>
                        <tr>{{ myResult.recommendations[1].model }}</tr>
                        <tr v-for="row in myResult.recommendations[1].roles" :key="row.role">
                            <tr> 
                                <td>Role</td>
                                <td>{{ row.role }}</td>  
                            </tr> 
                            <tr>
                                <td>Explanation</td>
                                <td>{{ row.explanation }}</td> 
                            </tr>
                            <tr>    
                                <td>Courses</td>
                                <tr v-for="course in row.courses">
                                    <tr>Title: {{ course.course }}</tr>
                                    <tr>Url: {{ course.url }}</tr>
                                    <tr>Explanation: {{ course.explanation }}</tr>
                                </tr>
                            </tr>                      
                        </tr>                    
                    </tbody>
                </table>
            </div>
            <div class="w-80" v-else-if="selected === 'Model-3'">
                <h2 class="text-2xl my-4  font-semibold">Mock</h2>
                <table class="table">
                    <tbody>
                        <tr>{{ myResult.recommendations[2].model }}</tr>
                        <tr v-for="row in myResult.recommendations[2].roles" :key="row.role">
                            <tr> 
                                <td>Role</td>
                                <td>{{ row.role }}</td>  
                            </tr> 
                            <tr>
                                <td>Explanation</td>
                                <td>{{ row.explanation }}</td> 
                            </tr>
                            <tr>    
                                <td>Courses</td>
                                <tr v-for="course in row.courses">
                                    <tr>Title: {{ course.course }}</tr>
                                    <tr>Url: {{ course.url }}</tr>
                                    <tr>Explanation: {{ course.explanation }}</tr>
                                </tr>
                            </tr>                      
                        </tr>                    
                    </tbody>
                    <!-- <tr>
                        <td>Role</td>
                        <td v-text="myResult.recommendations[0].role"></td>
                    </tr> -->
                </table>
            </div>
            <!-- <div class="space-x-4">
                <span>Role:</span>
                <span v-text="myResult.role"></span>
            </div> -->
            <!--<iframe src="https://docs.google.com/forms/d/e/1FAIpQLSdydUOOM0CvVoQUb6L4oCcmFxNKwbSOFzcBXK-jFFSodeBsYw/viewform?embedded=true" width="640" height="469" frameborder="0" marginheight="0" marginwidth="0">Yükleniyor…</iframe>
            -->

            <p class="mb-4">Please fill the
                <a class="underline text-blue-500" 
                    v-bind:href="'https://docs.google.com/forms/d/e/1FAIpQLSdydUOOM0CvVoQUb6L4oCcmFxNKwbSOFzcBXK-jFFSodeBsYw/viewform?entry.1655930766='+ myResult.fileName">
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